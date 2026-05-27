"""
attention_viz.py
----------------
Visualización de los mecanismos de atención de BART.

Tres tipos de atención en el Transformer:
  1. Encoder self-attention  — bidireccional, atiende todos los tokens
  2. Decoder masked self-attention — causal, solo tokens ya generados
  3. Cross-attention (decoder→encoder) — Q del decoder, K y V del encoder

Este módulo extrae y grafica los pesos de atención reales del modelo.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")           # backend sin GUI — compatible con Streamlit
import matplotlib.pyplot as plt
import io
from transformers import BartForConditionalGeneration, BartTokenizer


def _clean_tokens(tokens: list[str]) -> list[str]:
    """Limpia tokens BPE para visualización (elimina prefijos Ġ y tokens especiales)."""
    return [
        t.replace("\u0120", "")   # Ġ → espacio en BPE de BART
         .replace("</s>", "EOS")
         .replace("<s>",  "BOS")
        for t in tokens
    ]


def get_attention_weights(
    text: str,
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    device: str,
    max_length: int = 40,
):
    """
    Ejecuta un forward pass con output_attentions=True y retorna:
      - encoder_attentions : lista de tensores [n_heads, seq, seq] por capa
      - cross_attentions   : lista de tensores [n_heads, 1, seq] por capa
      - tokens             : lista de strings (tokens limpios)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # El decoder necesita al menos un token para calcular cross-attention
            decoder_input_ids=torch.tensor(
                [[model.config.decoder_start_token_id]]
            ).to(device),
            output_attentions=True,
        )

    tokens = _clean_tokens(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    )

    # Cada elemento: tensor [batch=1, n_heads, seq, seq] → quitamos batch dim
    encoder_attentions = [a[0] for a in outputs.encoder_attentions]
    cross_attentions   = [a[0] for a in outputs.cross_attentions]

    return encoder_attentions, cross_attentions, tokens


def plot_encoder_attention(
    encoder_attentions,
    tokens: list[str],
    layer: int = -1,
    max_tokens: int = 20,
) -> bytes:
    """
    Grafica los pesos de atención del encoder (última capa por defecto).
    Cada subplot = una cabeza de atención.

    En el encoder (bidireccional), Q, K y V se calculan así:
        Q = X @ W_Q   → "¿qué busca este token?"
        K = X @ W_K   → "¿qué información tiene este token?"
        V = X @ W_V   → "¿cuál es el contenido real?"
        scores = softmax(Q @ K.T / sqrt(d_k)) @ V

    Returns:
        bytes de imagen PNG (listo para st.image() en Streamlit)
    """
    attn = encoder_attentions[layer]          # [n_heads, seq, seq]
    n_heads  = min(attn.shape[0], 8)
    seq_len  = attn.shape[1]
    show     = min(max_tokens, seq_len)
    toks     = tokens[:show]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        f"Encoder Self-Attention — capa {layer} — primeras {n_heads} cabezas\n"
        "Cada cabeza aprende patrones distintos (sintaxis, semántica, correferencia...)",
        fontsize=12, fontweight="bold",
    )

    for idx, ax in enumerate(axes.flat):
        if idx >= n_heads:
            ax.axis("off")
            continue
        matrix = attn[idx].cpu().numpy()[:show, :show]
        im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=matrix.max())
        ax.set_title(f"Cabeza {idx + 1}", fontsize=9)
        ax.set_xticks(range(show))
        ax.set_xticklabels(toks, rotation=90, fontsize=6)
        ax.set_yticks(range(show))
        ax.set_yticklabels(toks, fontsize=6)

    plt.colorbar(im, ax=axes.flat[-1], fraction=0.046, label="Peso de atención")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_avg_attention(
    encoder_attentions,
    tokens: list[str],
    layer: int = -1,
    max_tokens: int = 20,
) -> bytes:
    """
    Grafica la atención promedio entre todas las cabezas del encoder.
    Muestra los patrones globales que el modelo considera más relevantes.

    Returns:
        bytes de imagen PNG
    """
    attn  = encoder_attentions[layer]         # [n_heads, seq, seq]
    show  = min(max_tokens, attn.shape[1])
    toks  = tokens[:show]
    avg   = attn.mean(dim=0).cpu().numpy()[:show, :show]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg, cmap="YlOrRd", aspect="auto")
    ax.set_title(
        "Atención Promedio del Encoder (todas las cabezas)\n"
        "Valores altos = mayor dependencia entre tokens",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(range(show))
    ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(show))
    ax.set_yticklabels(toks, fontsize=8)
    plt.colorbar(im, ax=ax, label="Peso de atención")
    ax.set_xlabel("Tokens Key (K) — ¿qué información tengo?", fontsize=9)
    ax.set_ylabel("Tokens Query (Q) — ¿qué busco?", fontsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_cross_attention(
    cross_attentions,
    tokens: list[str],
    layer: int = -1,
    max_tokens: int = 20,
) -> bytes:
    """
    Grafica la cross-attention del decoder sobre el encoder.

    En la cross-attention:
        Q  → viene del DECODER (el resumen que se está generando)
        K  → viene del ENCODER (representación del artículo original)
        V  → viene del ENCODER (representación del artículo original)

    Esto permite que cada token generado "mire" el artículo completo
    y decida qué partes son más relevantes para el siguiente token.

    Returns:
        bytes de imagen PNG
    """
    # cross_attentions[layer]: [n_heads, decoder_seq=1, encoder_seq]
    attn  = cross_attentions[layer]
    n_heads = min(attn.shape[0], 8)
    show    = min(max_tokens, attn.shape[-1])
    toks    = tokens[:show]

    fig, axes = plt.subplots(2, 4, figsize=(18, 5))
    fig.suptitle(
        f"Cross-Attention (Decoder→Encoder) — capa {layer}\n"
        "Q = decoder | K, V = encoder — el decoder 'lee' el artículo para generar el resumen",
        fontsize=12, fontweight="bold",
    )

    for idx, ax in enumerate(axes.flat):
        if idx >= n_heads:
            ax.axis("off")
            continue
        # shape: [decoder_seq=1, encoder_seq] → aplanamos
        vec = attn[idx, 0, :show].cpu().numpy().reshape(1, -1)
        im  = ax.imshow(vec, cmap="Greens", aspect="auto", vmin=0, vmax=vec.max())
        ax.set_title(f"Cabeza {idx + 1}", fontsize=9)
        ax.set_xticks(range(show))
        ax.set_xticklabels(toks, rotation=90, fontsize=6)
        ax.set_yticks([])

    plt.colorbar(im, ax=axes.flat[-1], fraction=0.046, label="Peso")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()