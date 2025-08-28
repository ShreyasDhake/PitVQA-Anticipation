import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, XCLIPModel
from peft import get_peft_model

class TemporalAttentionAdaptiveGating(nn.Module):
    """
    A temporal cross-attention mechanism that fuses video and text representations
    using an adaptive gate to control the flow of visual information.
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Temporal Encoder (GRU)
        self.video_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 2. Cross-Attention Module
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. Adaptive Gating Mechanism
        self.gating_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, frame_emb, text_emb):
        gru_out, _ = self.video_gru(frame_emb) # [B, T_frames, 2 * D]
        video_context = (gru_out[:, :, :self.embed_dim] + gru_out[:, :, self.embed_dim:]) # [B, T_frames, D]

        attn_output, _ = self.cross_attention(
            query=text_emb,
            key=video_context,
            value=video_context
        )

        gate = self.gating_layer(text_emb)
        gated_attn_output = gate * attn_output
        fused_emb = self.ln1(text_emb + gated_attn_output)

        ffn_output = self.ffn(fused_emb)
        final_output = self.ln2(fused_emb + ffn_output)

        return final_output

class PitVQAGen(nn.Module):
    def __init__(self, peft_config=None):
        super(PitVQAGen, self).__init__()

        model_name = "microsoft/xclip-base-patch32"
        self.visual_encoder = XCLIPModel.from_pretrained(model_name)
        self.video_proj = nn.Linear(self.visual_encoder.config.vision_config.hidden_size, 768)
        self.cross_attention_fusion = TemporalAttentionAdaptiveGating(embed_dim=768, num_heads=8, dropout=0.1)

        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_embedding = gpt2_model.transformer.wte
        self.gpt2_positional = gpt2_model.transformer.wpe
        self.gpt = get_peft_model(gpt2_model, peft_config)
        self.gpt.print_trainable_parameters()

    def forward(self, image, qa_inputs_ids, qa_att_mask):
        video = image.to(next(self.parameters()).device)
        batch_size, num_frames, C, H, W = video.shape
        
        video = video.view(batch_size * num_frames, C, H, W)
        frame_features = self.visual_encoder.vision_model(pixel_values=video).pooler_output
        frame_features = frame_features.view(batch_size, num_frames, -1) # Shape: [B, T_frames, 512]

        video_embeds = self.video_proj(frame_features) # Shape: [B, T_frames, 768]

        word_embeds = self.gpt2_embedding(qa_inputs_ids)
        pos_ids = torch.arange(qa_inputs_ids.shape[1], device=qa_inputs_ids.device).unsqueeze(0)
        pos_embeds = self.gpt2_positional(pos_ids)
        text_features = word_embeds + pos_embeds

        fused_text_features = self.cross_attention_fusion(video_embeds, text_features)
        
        gpt_output = self.gpt(
            inputs_embeds=fused_text_features,
            attention_mask=qa_att_mask
        )
        return gpt_output.logits
