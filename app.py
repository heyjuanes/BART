"""
app.py  —  Interfaz Streamlit para el proyecto BART
Uso: streamlit run app.py
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.model         import load_model, get_model_info
from src.inference     import summarize, summarize_batch
from src.attention_viz import (
    get_attention_weights, plot_encoder_attention,
    plot_avg_attention, plot_cross_attention,
)
from src.evaluation import EVAL_PAIRS, evaluate_batch

# ── Página ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BART Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general oscuro */
    .stApp { background-color: #0f1117; }

    /* Sidebar oscuro */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e2e3e;
    }

    /* Banner */
    .bart-banner {
        background: linear-gradient(160deg, #8b0000 0%, #c62828 60%, #7b0000 100%);
        border-radius: 10px;
        padding: 20px 18px 16px 18px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(183,28,28,0.4);
    }
    .bart-banner-title {
        color: #ffffff;
        font-size: 2.2em;
        font-weight: 800;
        letter-spacing: 6px;
        margin: 0 0 2px 0;
        font-family: 'Arial Black', 'Arial Bold', sans-serif;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        line-height: 1;
    }
    .bart-banner-sub {
        color: #ffcdd2;
        font-size: 0.72em;
        letter-spacing: 0.3px;
        margin: 0 0 14px 0;
        font-family: Arial, sans-serif;
        opacity: 0.8;
    }
    .bart-banner-links {
        display: flex;
        gap: 8px;
        margin-bottom: 14px;
        flex-wrap: wrap;
    }
    .bart-banner-links a {
        color: #fff;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.7em;
        text-decoration: none;
        font-family: Arial, sans-serif;
        letter-spacing: 0.3px;
        transition: background 0.2s;
    }
    .bart-banner-links a:hover { background: rgba(255,255,255,0.26); }
    .bart-banner-authors {
        color: #ffcdd2;
        font-size: 0.7em;
        border-top: 1px solid rgba(255,255,255,0.15);
        padding-top: 10px;
        line-height: 2;
        font-family: Arial, sans-serif;
        opacity: 0.85;
    }

    /* Cajas de contenido */
    .summary-box {
        background: #1e2235;
        border-left: 4px solid #ef5350;
        padding: 18px 20px;
        border-radius: 6px;
        font-size: 1.05em;
        line-height: 1.7;
        color: #e8eaf6 !important;
        margin-top: 10px;
    }
    .context-card {
        background: #1a1d27;
        border: 1px solid #2e2e3e;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 14px;
    }
    .context-card h4 {
        color: #ef5350;
        margin-top: 0;
        font-size: 1em;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .context-card p { color: #b0bec5; line-height: 1.7; }

    /* Textos generales */
    .stMarkdown p, .stMarkdown li { color: #cfd8dc; }
    h1, h2, h3 { color: #ffffff !important; }

    /* Radio buttons */
    [data-testid="stRadio"] label { color: #b0bec5 !important; }

    /* Metrics */
    [data-testid="stMetric"] { background: #1a1d27; border-radius: 8px; padding: 10px; }
    [data-testid="stMetricLabel"] { color: #90a4ae !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# ── Modelo ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo BART-large-CNN (~1.6 GB)...")
def get_model():
    return load_model()

model, tokenizer, device = get_model()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Banner lateral
    st.markdown("""
    <div class="bart-banner">
        <p class="bart-banner-title">BART</p>
        <p class="bart-banner-sub">Denoising Seq2Seq Pre-training · 406M params</p>
        <div class="bart-banner-links">
            <a href="https://huggingface.co/papers/1910.13461" target="_blank">📄 Artículo</a>
            <a href="https://huggingface.co/facebook/bart-large-cnn" target="_blank">🤗 Pesos</a>
        </div>
        <div class="bart-banner-authors">
            Juan Esteban Espitia<br>
            Daniel Fernando Mejía<br>
            Rubén Darío Salcedo
        </div>
    </div>
    """, unsafe_allow_html=True)

    section = st.radio(
        "Navegación",
        ["🧭 Contexto", "📝 Sumarización", "🔍 Atención", "📊 Evaluación ROUGE", "🏗️ Arquitectura"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"Dispositivo: **{device.upper()}**")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  0. CONTEXTO                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
if section == "🧭 Contexto":
    st.title("BART — Contexto del Proyecto")
    st.markdown("##### Procesamiento de Datos Secuenciales · Proyecto Final")
    st.divider()

    st.markdown("""
    <div class="context-card">
        <h4>🔴 Problemática</h4>
        <p>
        Los modelos de lenguaje tradicionales (LSTM, GRU, modelos extractivos) no logran
        resumir textos de forma <strong style="color:#ef9a9a">abstractiva, coherente y fiel al contenido original</strong>.
        Copian fragmentos del texto fuente sin reformulación real, fallan en capturar
        dependencias de largo alcance y no comprenden el contexto global del documento.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="context-card">
        <h4>📌 Artículo seleccionado</h4>
        <p>
        <strong style="color:#ef9a9a">BART: Denoising Sequence-to-Sequence Pre-training for Natural Language
        Generation, Translation, and Comprehension</strong><br>
        Lewis et al., 2019 ·
        <a href="https://huggingface.co/papers/1910.13461" style="color:#ef5350" target="_blank">Artículo</a>
        &nbsp;·&nbsp;
        <a href="https://huggingface.co/facebook/bart-large-cnn" style="color:#ef5350" target="_blank">Pesos preentrenados</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="context-card">
            <h4>🎯 Objetivo</h4>
            <p>
            Aplicar la arquitectura BART (Transformer encoder–decoder) a la tarea de
            <strong style="color:#ef9a9a">sumarización abstractiva</strong> de texto en inglés,
            implementando inferencia con pesos preentrenados y demostrando visualmente
            el mecanismo de atención.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="context-card">
            <h4>💡 Solución propuesta</h4>
            <p>
            Usar <strong style="color:#ef9a9a">facebook/bart-large-cnn</strong> — BART-Large
            fine-tuneado en CNN/DailyMail — para inferencia directa sin reentrenamiento.
            El modelo recibe un artículo en inglés y genera un resumen abstractivo
            mediante beam search con 4 hipótesis paralelas.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="context-card">
        <h4>🧠 ¿Por qué BART resuelve el problema?</h4>
        <p>
        BART combina un <strong style="color:#ef9a9a">encoder bidireccional</strong> (entiende el artículo completo,
        como BERT) con un <strong style="color:#ef9a9a">decoder autoregresivo</strong> (genera texto nuevo, como GPT).
        El preentrenamiento con múltiples funciones de ruido — especialmente <em>Text Infilling</em>
        y <em>Sentence Permutation</em> — obliga al modelo a reconstruir texto corrupto,
        desarrollando una comprensión profunda de la semántica del lenguaje.
        La <strong style="color:#ef9a9a">cross-attention</strong> (Q del decoder, K y V del encoder) permite que
        cada token generado consulte el artículo original completo, garantizando
        coherencia y fidelidad al contenido.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="context-card">
        <h4>📦 Stack tecnológico</h4>
        <p>
        🤗 HuggingFace Transformers 4.40 &nbsp;·&nbsp;
        🔥 PyTorch &nbsp;·&nbsp;
        📊 ROUGE-score &nbsp;·&nbsp;
        🌐 Streamlit &nbsp;·&nbsp;
        📈 Matplotlib
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<p style='color:#546e7a; font-size:0.85em; text-align:center'>"
        "Juan Esteban Espitia · Daniel Fernando Mejía · Rubén Darío Salcedo</p>",
        unsafe_allow_html=True,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. SUMARIZACIÓN                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
elif section == "📝 Sumarización":
    st.title("Sumarización Abstractiva")
    st.markdown(
        "<p style='color:#b0bec5'>BART genera resúmenes <strong>abstractivos</strong> — no copia frases del original, "
        "sino que las reformula usando lo aprendido durante el preentrenamiento denoising.</p>",
        unsafe_allow_html=True,
    )

    texto = st.text_area(
        "Artículo en inglés (mínimo 20 palabras):",
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
            c1.metric("Palabras originales", result["input_words"])
            c2.metric("Palabras resumen",    result["output_words"])
            c3.metric("Compresión",         f'{result["compression_pct"]}%')
            c4.metric("Tokens de entrada",  result["input_tokens"])

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
    st.title("Mecanismo de Atención")
    st.markdown("""
<p style='color:#b0bec5'>
<strong>Fórmula de atención multi-cabeza:</strong>
</p>
""", unsafe_allow_html=True)
    st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V")
    st.markdown(
        "<p style='color:#90a4ae; font-size:0.9em'>"
        "<code>Q = X @ W_Q</code> &nbsp;·&nbsp; <code>K = X @ W_K</code> &nbsp;·&nbsp; <code>V = X @ W_V</code>"
        " — proyecciones lineales de la entrada</p>",
        unsafe_allow_html=True,
    )

    texto_viz = st.text_area(
        "Texto para visualizar (inglés, ~30 palabras para mejor visualización):",
        value=(
            "Deforestation threatens the Amazon rainforest, a critical carbon sink. "
            "Governments pledged to reduce logging but enforcement remains weak."
        ),
        height=100,
    )

    tipo = st.radio(
        "Tipo de atención:",
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
                "Q y K del encoder — mismo tensor proyectado con W_Q y W_K. "
                "Cada cabeza captura patrones distintos: sintaxis, semántica, correferencia."
            )
        elif tipo == "Encoder — promedio global":
            img = plot_avg_attention(enc_attn, tokens)
            st.image(img, use_container_width=True)
            st.caption(
                "Promedio de las 16 cabezas. Valores altos = mayor dependencia entre tokens."
            )
        else:
            img = plot_cross_attention(cross_attn, tokens)
            st.image(img, use_container_width=True)
            st.caption(
                "Cross-attention: Q viene del DECODER, K y V del ENCODER. "
                "Muestra a qué partes del artículo 'mira' el decoder al generar el resumen."
            )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. EVALUACIÓN ROUGE                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
elif section == "📊 Evaluación ROUGE":
    st.title("Evaluación con Métricas ROUGE")
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
            "Las diferencias vs benchmark son normales: el paper evalúa sobre 11.490 artículos "
            "del test set de CNN/DailyMail. Nuestras 3 muestras tienen referencias distintas al estilo de ese dataset."
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. ARQUITECTURA                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
elif section == "🏗️ Arquitectura":
    st.title("Arquitectura de BART-Large")

    info = get_model_info(model)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Encoder (bidireccional)")
        st.table({
            "Parámetro": ["Capas", "d_model", "Cabezas atención", "d_k por cabeza",
                          "FFN interno", "Q proj shape", "K proj shape", "V proj shape"],
            "Valor": [
                info["encoder_layers"], info["d_model"], info["attention_heads"],
                info["d_k_per_head"], info["ffn_dim"],
                str(info["encoder_q_proj"]), str(info["encoder_k_proj"]), str(info["encoder_v_proj"]),
            ]
        })
    with col2:
        st.markdown("#### Decoder (autoregresivo) — cross-attention")
        st.table({
            "Parámetro": ["Capas", "d_model", "Cabezas atención", "Cross-Attn Q (decoder)",
                          "Cross-Attn K (encoder)", "Cross-Attn V (encoder)", "Vocab size"],
            "Valor": [
                info["decoder_layers"], info["d_model"], info["attention_heads"],
                str(info["cross_attn_q_proj"]), str(info["cross_attn_k_proj"]),
                str(info["cross_attn_v_proj"]), f"{info['vocab_size']:,}",
            ]
        })

    st.markdown("#### Distribución de parámetros")
    total = info["total_params_M"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Encoder",  f"{info['encoder_params_M']:.1f}M", f"{100*info['encoder_params_M']/total:.0f}%")
    c2.metric("Decoder",  f"{info['decoder_params_M']:.1f}M", f"{100*info['decoder_params_M']/total:.0f}%")
    c3.metric("LM Head",  f"{info['lm_head_params_M']:.1f}M", f"{100*info['lm_head_params_M']/total:.0f}%")
    c4.metric("Total",    f"{total:.1f}M")

    st.markdown("#### BART vs modelos anteriores")
    st.table({
        "Característica": ["Encoder", "Decoder", "Pre-entrenamiento", "Generación", "Comprensión", "Sumarización"],
        "BERT":  ["Bidireccional", "No tiene",      "Masked LM",         "Limitada",  "Excelente", "No"],
        "GPT-2": ["No tiene",      "Autoregresivo", "Causal LM",         "Buena",     "Limitada",  "No"],
        "BART":  ["Bidireccional", "Autoregresivo", "Denoising (multi)", "Excelente", "Buena",     "Estado del arte"],
    })

    with st.expander("¿Por qué BART es innovador?"):
        st.markdown("""
1. **Une lo mejor de BERT y GPT:** encoder bidireccional + decoder autoregresivo en un solo modelo.
2. **Preentrenamiento denoising más flexible:** *Text Infilling* usa un solo `[MASK]` para spans completos — más difícil que el masking individual de BERT.
3. **Un modelo, múltiples tareas:** sumarización, traducción, comprensión y generación con los mismos pesos.
4. **Cross-attention como puente:** Q del decoder, K/V del encoder — el decoder consulta el artículo completo en cada paso de generación.
        """)