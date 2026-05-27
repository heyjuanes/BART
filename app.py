"""
app.py
------
Interfaz interactiva con Streamlit para el proyecto BART.

Uso:
    streamlit run app.py

Secciones:
    1. Sumarización       — ingresa cualquier artículo y obtén el resumen
    2. Atención           — visualiza encoder self-attention y cross-attention
    3. Evaluación ROUGE   — compara contra benchmark del paper
    4. Arquitectura       — info estructural del modelo
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.model      import load_model, get_model_info
from src.inference  import summarize, summarize_batch
from src.attention_viz import (
    get_attention_weights,
    plot_encoder_attention,
    plot_avg_attention,
    plot_cross_attention,
)
from src.evaluation import EVAL_PAIRS, evaluate_batch, format_rouge_table

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="BART Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS mínimo ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: #f0f4ff;
        border-left: 4px solid #1565C0;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .summary-box {
        background: #f8f9fa;
        border-left: 4px solid #2e7d32;
        padding: 16px;
        border-radius: 4px;
        font-size: 1.05em;
        line-height: 1.6;
    }
    .rouge-good  { color: #2e7d32; font-weight: bold; }
    .rouge-ok    { color: #f57c00; font-weight: bold; }
    .rouge-low   { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Carga del modelo (cacheada — solo una vez por sesión) ─────────────────────
@st.cache_resource(show_spinner="Cargando modelo BART-large-CNN (~1.6 GB)...")
def get_model():
    return load_model()


model, tokenizer, device = get_model()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=40)
    st.title("BART Summarizer")
    st.caption("facebook/bart-large-cnn · ~400M params")
    st.divider()

    section = st.radio(
        "Sección",
        ["📝 Sumarización", "🔍 Atención", "📊 Evaluación ROUGE", "🏗️ Arquitectura"],
    )

    st.divider()
    st.caption(f"Dispositivo: **{device.upper()}**")
    st.caption("Paper: [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)")
    st.caption("Modelo: [HuggingFace](https://huggingface.co/facebook/bart-large-cnn)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. SUMARIZACIÓN                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
if section == "📝 Sumarización":
    st.header("Sumarización Abstractiva con BART")
    st.markdown(
        "BART genera resúmenes **abstractivos** — no copia frases del original, "
        "sino que las reformula usando lo aprendido durante el preentrenamiento denoising."
    )

    texto = st.text_area(
        "Artículo (en inglés, mínimo 40 palabras):",
        height=220,
        placeholder=(
            "Researchers at MIT have developed a new artificial intelligence system capable "
            "of detecting early signs of Alzheimer's disease up to six years before a clinical "
            "diagnosis. The model analyzes speech patterns and linguistic features from routine "
            "conversations, identifying subtle changes in vocabulary complexity, sentence structure, "
            "and word-finding pauses that often precede cognitive decline..."
        ),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        max_len = st.slider("Longitud máxima (tokens)", 50, 300, 130, 10)
    with col2:
        min_len = st.slider("Longitud mínima (tokens)", 10, 100, 30, 5)
    with col3:
        beams = st.slider("Beam search — nº de beams", 1, 6, 4)

    if st.button("▶ Generar resumen", type="primary"):
        if len(texto.split()) < 20:
            st.warning("Por favor ingresa un texto más largo (mínimo 20 palabras).")
        else:
            with st.spinner("Generando resumen..."):
                result = summarize(texto, model, tokenizer, device,
                                   max_length=max_len, min_length=min_len, num_beams=beams)

            st.subheader("Resumen generado")
            st.markdown(
                f'<div class="summary-box">{result["summary"]}</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Palabras originales",  result["input_words"])
            c2.metric("Palabras resumen",      result["output_words"])
            c3.metric("Compresión",           f'{result["compression_pct"]}%')
            c4.metric("Tokens de entrada",    result["input_tokens"])

            with st.expander("ℹ️ ¿Cómo funciona internamente?"):
                st.markdown(f"""
**Proceso de inferencia paso a paso:**

1. **Tokenización BPE** → el texto se convierte en `{result["input_tokens"]}` tokens usando Byte-Pair Encoding  
2. **Encoder bidireccional** → procesa TODOS los tokens simultáneamente con self-attention (sin máscara causal)  
   - Genera `hidden_states` de forma `[1, {result["input_tokens"]}, 1024]`  
3. **Decoder autoregresivo** → genera token a token usando:  
   - **Masked self-attention** — solo ve tokens ya generados  
   - **Cross-attention** — Q del decoder, K y V del encoder (así "lee" el artículo)  
4. **Beam search con {beams} beams** — mantiene las {beams} mejores hipótesis en paralelo  
5. **Decodificación** → `{result["output_tokens"]}` tokens → texto final  
                """)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2. ATENCIÓN                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
elif section == "🔍 Atención":
    st.header("Visualización del Mecanismo de Atención")
    st.markdown("""
**Fórmula de atención multi-cabeza:**

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$$

Donde `Q = X @ W_Q`, `K = X @ W_K`, `V = X @ W_V` son proyecciones lineales de la entrada.
    """)

    texto_viz = st.text_area(
        "Texto para visualizar (inglés, máx ~30 palabras para mejor visualización):",
        value=(
            "Deforestation threatens the Amazon rainforest, a critical carbon sink. "
            "Governments pledged to reduce logging but enforcement remains weak."
        ),
        height=100,
    )

    tipo = st.radio(
        "Tipo de atención a visualizar:",
        ["Encoder self-attention (por cabeza)", "Encoder — promedio global", "Cross-attention (decoder→encoder)"],
        horizontal=True,
    )

    if st.button("▶ Visualizar atención", type="primary"):
        with st.spinner("Extrayendo pesos de atención..."):
            enc_attn, cross_attn, tokens = get_attention_weights(
                texto_viz, model, tokenizer, device
            )

        if tipo == "Encoder self-attention (por cabeza)":
            img = plot_encoder_attention(enc_attn, tokens)
            st.image(img, use_container_width=True)
            st.caption(
                "**Q y K del encoder** — mismo tensor de entrada proyectado con W_Q y W_K. "
                "Cada cabeza captura patrones distintos: algunas se especializan en sintaxis, "
                "otras en semántica, otras en correferencia."
            )

        elif tipo == "Encoder — promedio global":
            img = plot_avg_attention(enc_attn, tokens)
            st.image(img, use_container_width=True)
            st.caption(
                "Promedio de las 16 cabezas. Los valores altos indican pares de tokens "
                "con alta dependencia — el modelo considera que esos tokens se 'necesitan' mutuamente."
            )

        else:
            img = plot_cross_attention(cross_attn, tokens)
            st.image(img, use_container_width=True)
            st.caption(
                "**Cross-attention:** Q viene del DECODER (token siendo generado), "
                "K y V vienen del ENCODER (artículo original). "
                "Muestra a qué partes del artículo 'mira' el decoder al generar el resumen."
            )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. EVALUACIÓN ROUGE                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
elif section == "📊 Evaluación ROUGE":
    st.header("Evaluación con Métricas ROUGE")
    st.markdown("""
| Métrica | Descripción |
|---------|-------------|
| **ROUGE-1** | Solapamiento de unigramas (palabras individuales) |
| **ROUGE-2** | Solapamiento de bigramas (pares consecutivos) |
| **ROUGE-L** | Subsecuencia común más larga (captura orden) |

**Benchmark oficial** en CNN/DailyMail: ROUGE-1=42.95 · ROUGE-2=20.82 · ROUGE-L=30.62
    """)

    if st.button("▶ Ejecutar evaluación (3 muestras)", type="primary"):
        articulos = [p["articulo"] for p in EVAL_PAIRS]

        with st.spinner("Generando resúmenes y calculando ROUGE..."):
            batch_results = summarize_batch(articulos, model, tokenizer, device)
            summaries     = [r["summary"] for r in batch_results]
            eval_results  = evaluate_batch(summaries)

        st.subheader("Resultados por muestra")
        for i, (par, summ, scores) in enumerate(
            zip(EVAL_PAIRS, summaries, eval_results["per_sample"])
        ):
            with st.expander(f"Muestra #{i+1}"):
                st.markdown(f"**Artículo:** {par['articulo'][:200]}...")
                st.markdown(f"**Referencia humana:** {par['referencia']}")
                st.markdown(
                    f'<div class="summary-box"><b>BART:</b> {summ}</div>',
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("ROUGE-1", f"{scores['rouge1']:.3f}")
                c2.metric("ROUGE-2", f"{scores['rouge2']:.3f}")
                c3.metric("ROUGE-L", f"{scores['rougeL']:.3f}")

        st.subheader("Promedio vs Benchmark")
        avg = eval_results["averages"]
        bm  = eval_results["benchmark"]
        c1, c2, c3 = st.columns(3)
        c1.metric("ROUGE-1", f"{avg['rouge1']:.3f}", f"{avg['rouge1']-bm['rouge1']:+.3f} vs paper")
        c2.metric("ROUGE-2", f"{avg['rouge2']:.3f}", f"{avg['rouge2']-bm['rouge2']:+.3f} vs paper")
        c3.metric("ROUGE-L", f"{avg['rougeL']:.3f}", f"{avg['rougeL']-bm['rougeL']:+.3f} vs paper")

        st.caption(
            "Las diferencias vs benchmark son normales: el paper evalúa en el test set completo "
            "de CNN/DailyMail (11.490 artículos). Nuestras 3 muestras pueden tener referencias "
            "distintas al estilo de ese dataset."
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. ARQUITECTURA                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
elif section == "🏗️ Arquitectura":
    st.header("Arquitectura de BART-Large")

    info = get_model_info(model)

    st.markdown("### Estructura general")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Encoder (bidireccional)**")
        st.table({
            "Parámetro": ["Capas", "d_model", "Cabezas atención", "d_k por cabeza", "FFN interno",
                          "Q proj shape", "K proj shape", "V proj shape"],
            "Valor": [
                info["encoder_layers"], info["d_model"], info["attention_heads"],
                info["d_k_per_head"], info["ffn_dim"],
                str(info["encoder_q_proj"]), str(info["encoder_k_proj"]), str(info["encoder_v_proj"]),
            ]
        })

    with col2:
        st.markdown("**Decoder (autoregresivo) — cross-attention**")
        st.table({
            "Parámetro": ["Capas", "d_model", "Cabezas atención", "Cross-Attn Q (decoder)",
                          "Cross-Attn K (encoder)", "Cross-Attn V (encoder)", "Vocab size"],
            "Valor": [
                info["decoder_layers"], info["d_model"], info["attention_heads"],
                str(info["cross_attn_q_proj"]), str(info["cross_attn_k_proj"]),
                str(info["cross_attn_v_proj"]), f"{info['vocab_size']:,}",
            ]
        })

    st.markdown("### Distribución de parámetros")
    total = info["total_params_M"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Encoder",  f"{info['encoder_params_M']:.1f}M", f"{100*info['encoder_params_M']/total:.0f}%")
    c2.metric("Decoder",  f"{info['decoder_params_M']:.1f}M", f"{100*info['decoder_params_M']/total:.0f}%")
    c3.metric("LM Head",  f"{info['lm_head_params_M']:.1f}M", f"{100*info['lm_head_params_M']/total:.0f}%")
    c4.metric("Total",    f"{total:.1f}M")

    st.markdown("### Innovaciones vs modelos anteriores")
    st.table({
        "Característica": ["Encoder", "Decoder", "Pre-entrenamiento", "Generación", "Comprensión", "Sumarización"],
        "BERT":   ["Bidireccional", "No tiene",      "Masked LM",        "Limitada",   "Excelente",  "No"],
        "GPT-2":  ["No tiene",      "Autoregresivo", "Causal LM",        "Buena",      "Limitada",   "No"],
        "BART":   ["Bidireccional", "Autoregresivo", "Denoising (multi)","Excelente",  "Buena",      "Estado del arte"],
    })

    with st.expander("¿Por qué BART es innovador?"):
        st.markdown("""
1. **Une lo mejor de BERT y GPT:** encoder bidireccional + decoder autoregresivo en un solo modelo.
2. **Preentrenamiento denoising más flexible:** en lugar del masking uniforme de BERT, BART aplica
   múltiples funciones de ruido. La más importante para sumarización es *Text Infilling*: un solo
   `[MASK]` reemplaza spans completos de longitud variable — mucho más difícil que el masking individual.
3. **Un modelo, múltiples tareas:** con fine-tuning, el mismo modelo sirve para sumarización,
   traducción, comprensión y generación. BERT necesita cabezas distintas y no genera texto;
   GPT-2 no comprende bidireccionalmente.
4. **Cross-attention como puente:** el decoder puede atender CUALQUIER parte del artículo original
   en cada paso de generación gracias a la cross-attention (Q del decoder, K/V del encoder).
        """)