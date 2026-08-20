import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, XCLIPModel
from peft import get_peft_model, LoraConfig, TaskType


LANGUAGE_MODEL_CONFIGS = {
    "gpt2": {
        "pretrained_name": "gpt2",
        "artifact_prefix": "GPT2",
        "lora_target_modules": ["c_attn", "c_proj"],
    },
    "qwen": {
        "pretrained_name": "Qwen/Qwen3-0.6B",
        "artifact_prefix": "Qwen",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj"],
    },
}


def get_language_model_config(language_model):
    try:
        return LANGUAGE_MODEL_CONFIGS[language_model]
    except KeyError as error:
        choices = ", ".join(LANGUAGE_MODEL_CONFIGS)
        raise ValueError(
            f"Unsupported language model '{language_model}'. Choose one of: {choices}."
        ) from error


def create_tokenizer(language_model):
    config = get_language_model_config(language_model)
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_name"])
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_lora_config(language_model, dropout=0.1):
    config = get_language_model_config(language_model)
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=dropout,
        target_modules=config["lora_target_modules"],
    )


def get_artifact_prefix(language_model):
    return get_language_model_config(language_model)["artifact_prefix"]


def add_artifact_prefix(filename, language_model):
    prefix = get_artifact_prefix(language_model)
    if filename.lower().startswith(f"{prefix.lower()}_"):
        return filename
    return f"{prefix}_{filename}"

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
    def __init__(self, language_model="gpt2", peft_config=None, dropout=0.1):
        super(PitVQAGen, self).__init__()

        language_config = get_language_model_config(language_model)
        self.language_model_name = language_model

        model_name = "microsoft/xclip-base-patch32"
        self.visual_encoder = XCLIPModel.from_pretrained(model_name)

        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        self.tokenizer = create_tokenizer(language_model)
        causal_lm = AutoModelForCausalLM.from_pretrained(
            language_config["pretrained_name"]
        )
        hidden_size = causal_lm.config.hidden_size
        vision_hidden_size = self.visual_encoder.config.vision_config.hidden_size

        self.video_proj = nn.Linear(vision_hidden_size, hidden_size)
        self.cross_attention_fusion = TemporalAttentionAdaptiveGating(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
        )
        if language_model == "gpt2":
            # Keep the original attribute names so existing GPT-2 checkpoints
            # remain loadable.
            self.gpt2_embedding = causal_lm.transformer.wte
            self.gpt2_positional = causal_lm.transformer.wpe
            self.gpt = get_peft_model(causal_lm, peft_config)
            self.gpt.print_trainable_parameters()
        else:
            # Keep the original Qwen attribute name for the same reason.
            self.qwen = get_peft_model(causal_lm, peft_config)
            self.qwen.print_trainable_parameters()

    def forward(self, image, qa_inputs_ids, qa_att_mask):
        video = image.to(next(self.parameters()).device)
        batch_size, num_frames, C, H, W = video.shape
        
        video = video.view(batch_size * num_frames, C, H, W)
        frame_features = self.visual_encoder.vision_model(pixel_values=video).pooler_output
        frame_features = frame_features.view(batch_size, num_frames, -1) # Shape: [B, T_frames, 512]

        video_embeds = self.video_proj(frame_features) # Shape: [B, T_frames, 768]

        if self.language_model_name == "gpt2":
            text_features = self.gpt2_embedding(qa_inputs_ids)
            pos_ids = torch.arange(
                qa_inputs_ids.shape[1], device=qa_inputs_ids.device
            ).unsqueeze(0)
            text_features = text_features + self.gpt2_positional(pos_ids)
            decoder = self.gpt
        else:
            # Qwen applies rotary positional information inside its attention
            # layers, so only token embeddings are fused here.
            text_features = self.qwen.get_input_embeddings()(qa_inputs_ids)
            decoder = self.qwen

        fused_text_features = self.cross_attention_fusion(video_embeds, text_features)
        
        language_output = decoder(
            inputs_embeds=fused_text_features,
            attention_mask=qa_att_mask
        )
        return language_output.logits
