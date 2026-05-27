"""
model.py
--------
Carga del modelo BART y tokenizer desde Hugging Face.
Modelo: facebook/bart-large-cnn (~1.6 GB, descarga automática en primer uso)
"""

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

MODEL_NAME = "facebook/bart-large-cnn"


def get_device() -> str:
    """Retorna 'cuda' si hay GPU disponible, si no 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device: str = None):
    """
    Descarga (si no existe en caché) y carga el modelo BART-large-CNN.

    Args:
        device: 'cuda' o 'cpu'. Si es None, se detecta automáticamente.

    Returns:
        model: BartForConditionalGeneration en modo eval
        tokenizer: BartTokenizer correspondiente
        device: string del dispositivo usado
    """
    if device is None:
        device = get_device()

    print(f"[model] Dispositivo: {device}")
    print(f"[model] Cargando tokenizer '{MODEL_NAME}'...")
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

    print(f"[model] Cargando modelo '{MODEL_NAME}'...")
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()  # desactiva dropout, no necesitamos gradientes para inferencia

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Modelo listo — {total_params / 1e6:.1f}M parámetros")
    print(f"[model]   Encoder layers : {model.config.encoder_layers}")
    print(f"[model]   Decoder layers : {model.config.decoder_layers}")
    print(f"[model]   d_model        : {model.config.d_model}")
    print(f"[model]   Attn heads     : {model.config.encoder_attention_heads}")
    print(f"[model]   Vocab size     : {model.config.vocab_size:,}")

    return model, tokenizer, device


def get_model_info(model) -> dict:
    """
    Retorna un diccionario con información estructural del modelo.
    Útil para mostrar en la interfaz o en el README.
    """
    enc   = model.model.encoder
    dec   = model.model.decoder
    enc_l = enc.layers[0]
    dec_l = dec.layers[0]

    return {
        "encoder_layers"      : len(enc.layers),
        "decoder_layers"      : len(dec.layers),
        "d_model"             : model.config.d_model,
        "attention_heads"     : model.config.encoder_attention_heads,
        "d_k_per_head"        : enc_l.self_attn.head_dim,
        "ffn_dim"             : enc_l.fc1.out_features,
        "vocab_size"          : model.config.vocab_size,
        # Shapes de proyecciones Q, K, V en el encoder
        "encoder_q_proj"      : tuple(enc_l.self_attn.q_proj.weight.shape),
        "encoder_k_proj"      : tuple(enc_l.self_attn.k_proj.weight.shape),
        "encoder_v_proj"      : tuple(enc_l.self_attn.v_proj.weight.shape),
        # Cross-attention en el decoder (Q del decoder, K/V del encoder)
        "cross_attn_q_proj"   : tuple(dec_l.encoder_attn.q_proj.weight.shape),
        "cross_attn_k_proj"   : tuple(dec_l.encoder_attn.k_proj.weight.shape),
        "cross_attn_v_proj"   : tuple(dec_l.encoder_attn.v_proj.weight.shape),
        # Distribución de parámetros
        "encoder_params_M"    : sum(p.numel() for p in enc.parameters()) / 1e6,
        "decoder_params_M"    : sum(p.numel() for p in dec.parameters()) / 1e6,
        "lm_head_params_M"    : sum(p.numel() for p in model.lm_head.parameters()) / 1e6,
        "total_params_M"      : sum(p.numel() for p in model.parameters()) / 1e6,
    }