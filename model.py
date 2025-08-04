import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, XCLIPModel
from peft import get_peft_model

# GRU_Gated
class GRU_Gated(nn.Module):
    """Temporal cross‑attention with an adaptive gate between video and text."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Bidirectional GRU encodes temporal context
        self.video_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 2. Cross‑attention (text queries video)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 3. Adaptive gate – decides how much to mix the attended video info
        self.gating_layer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

        # standard transformer refinements
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, frame_emb, text_emb):
        """frame_emb: [B, T, D]  |  text_emb: [B, L, D]"""
        # Bidirectional GRU
        gru_out, _ = self.video_gru(frame_emb)                          # [B, T, 2D]
        video_ctx = gru_out[:, :, : self.embed_dim] + gru_out[:, :, self.embed_dim :]  # [B, T, D]

        # Cross‑attention: text queries video
        attn_out, _ = self.cross_attention(query=text_emb, key=video_ctx, value=video_ctx)

        # Gated fusion + residual
        gate = self.gating_layer(text_emb)
        fused = self.ln1(text_emb + gate * attn_out)

        # Feed‑forward + residual
        ffn_out = self.ffn(fused)
        return self.ln2(fused + ffn_out)

# PitVQAGen
class PitVQAGen(nn.Module):
    """Video‑conditioned GPT‑2 generator for surgical VQA."""

    def __init__(self, peft_config=None, vision_model_name: str = "microsoft/xclip-base-patch32"):
        super().__init__()

        # Vision encoder (frozen)
        self.visual_encoder = XCLIPModel.from_pretrained(vision_model_name)
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

        self.video_proj = nn.Linear(self.visual_encoder.config.vision_config.hidden_size, 768)

        # Cross‑modal fusion
        self.cross_attention_fusion = GRU_Gated(embed_dim=768)

        # GPT‑2 backbone with LoRA
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt2_embedding = gpt2.transformer.wte
        self.gpt2_positional = gpt2.transformer.wpe

        # convert to PEFT/LoRA
        self.gpt = get_peft_model(gpt2, peft_config) if peft_config else gpt2
        if hasattr(self.gpt, "print_trainable_parameters"):
            self.gpt.print_trainable_parameters()

    def forward(self, image, qa_inputs_ids, qa_att_mask):
        """image: [B, T, 3, 224, 224]  |  qa_inputs_ids: [B, L]"""
        device = next(self.parameters()).device
        video = image.to(device)
        B, T, C, H, W = video.shape

        # encode each frame separately
        video = video.view(B * T, C, H, W)
        frame_feat = self.visual_encoder.vision_model(pixel_values=video).pooler_output
        frame_feat = frame_feat.view(B, T, -1)  # [B, T, 512]

        video_emb = self.video_proj(frame_feat)  # [B, T, 768]

        # text embedding from GPT‑2
        word_emb = self.gpt2_embedding(qa_inputs_ids)
        pos_ids = torch.arange(qa_inputs_ids.size(1), device=device).unsqueeze(0)
        pos_emb = self.gpt2_positional(pos_ids)
        text_feat = word_emb + pos_emb

        # cross‑modal fusion
        fused_text = self.cross_attention_fusion(video_emb, text_feat)

        # LM head
        output = self.gpt(inputs_embeds=fused_text, attention_mask=qa_att_mask)
        return output.logits
