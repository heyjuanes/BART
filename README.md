# BART — Abstractive Summarization with `facebook/bart-large-cnn`

> **Procesamiento de Datos Secuenciales · Proyecto Final**
> Juan Esteban Espitia · Daniel Fernando Mejía · Rubén Darío Salcedo

Implementación de inferencia y visualización de atención sobre el modelo preentrenado **BART-large-CNN** (406 M parámetros) para sumarización abstractiva en inglés. Basado en el paper de Lewis et al., 2019 — [arXiv:1910.13461](https://arxiv.org/abs/1910.13461).

---

## El problema que resuelve

Los modelos tradicionales (LSTM, GRU) no logran generar resúmenes abstractivos coherentes:
no capturan dependencias de largo alcance, repiten frases del texto original sin reformulación
real, y no comprenden el contexto global del documento. BART resuelve esto con un encoder
bidireccional que lee el artículo completo antes de que el decoder autoregresivo genere
el resumen con sus propias palabras.

---

## Demo rápida

```bash
# Instalar dependencias
pip install -r requirements.txt

# Lanzar la interfaz
streamlit run app.py
```

La interfaz corre en CPU (~10–20 s por inferencia). No se requiere GPU.

---

## Estructura del proyecto

```
BART/
├── app.py                # Interfaz Streamlit — 5 secciones interactivas
├── requirements.txt
└── src/
    ├── model.py          # Carga del modelo y tokenizer; info de arquitectura
    ├── inference.py      # Sumarización con beam search; tokenización BPE
    ├── attention_viz.py  # Visualización de encoder self-attention y cross-attention
    └── evaluation.py     # Métricas ROUGE con 3 pares de prueba y benchmark del paper
```

---

## Arquitectura de BART-large

| Parámetro | Valor |
|---|---|
| Capas encoder / decoder | 12 / 12 |
| Dimensión oculta (`d_model`) | 1 024 |
| Cabezas de atención | 16 |
| `d_k` por cabeza | 64 (16 × 64 = 1 024) |
| FFN interno | 4 096 |
| Vocabulario BPE | 50 265 tokens |
| Parámetros totales | ~406 M |

BART combina un **encoder bidireccional** (como BERT) con un **decoder autoregresivo** (como GPT).
El preentrenamiento mediante *denoising* obliga al modelo a reconstruir texto a partir de versiones
corrompidas, desarrollando comprensión semántica profunda.

### Los tres tipos de atención en BART

```
Encoder self-attention
  Q, K, V ← mismo tensor del encoder
  Bidireccional: cada token atiende a todos los demás sin restricción

Decoder masked self-attention
  Q, K, V ← tensor del decoder
  Máscara causal: el token en posición i no puede ver posiciones futuras

Cross-attention  (decoder → encoder)
  Q ← decoder  |  K, V ← encoder
  Cada token del resumen consulta el artículo completo en cada paso de generación
```

La fórmula de atención es:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

Dividir por `√d_k = √64 = 8` evita que el softmax se sature con valores grandes,
lo que preserva gradientes útiles durante el entrenamiento.

---

## Tokenización BPE

El tokenizer usa *Byte-Pair Encoding* con vocabulario fijo de 50 265 tokens.
BPE descompone palabras desconocidas en subpalabras conocidas:

```
"deforestation" → ["de", "forest", "ation"]
```

Tokens especiales que BART agrega automáticamente:
- `<s>` (BOS, ID=0) — inicio de secuencia
- `</s>` (EOS, ID=2) — fin de secuencia

Longitud máxima de entrada: **1 024 tokens** (`truncation=True`).

---

## Generación con beam search

```python
model.generate(
    input_ids,
    num_beams=4,          # 4 hipótesis en paralelo
    max_length=130,
    min_length=30,
    length_penalty=2.0,   # > 1.0 favorece resúmenes más largos
    no_repeat_ngram_size=3,
    early_stopping=True,
)
```

Beam search mantiene las 4 mejores hipótesis simultáneamente, evitando la solución greedy
subóptima que se obtendría eligiendo el token más probable en cada paso de forma independiente.

---

## Preentrenamiento denoising

BART se preentrenó con cinco funciones de corrupción:

| Función | Descripción |
|---|---|
| Token Masking | Reemplaza tokens aleatorios con `[MASK]` (igual que BERT) |
| Token Deletion | Elimina tokens completamente; el modelo no sabe cuántos ni dónde |
| **Text Infilling** | Reemplaza spans de longitud variable por un solo `[MASK]`; el modelo infiere cuántas palabras faltan — **la más importante para sumarización** |
| Sentence Permutation | Mezcla el orden de las oraciones; el modelo reconstruye el orden correcto |
| Document Rotation | Rota desde un token aleatorio; el modelo identifica el inicio real |

La diferencia clave respecto a BERT: *Text Infilling* obliga al modelo a inferir **cuántos** tokens
faltan, no solo cuáles, desarrollando comprensión semántica global del documento.

---

## Evaluación ROUGE

ROUGE (*Recall-Oriented Understudy for Gisting Evaluation*) mide solapamiento entre el resumen
generado y uno de referencia humana:

- **ROUGE-1**: solapamiento de unigramas (palabras individuales)
- **ROUGE-2**: solapamiento de bigramas (pares consecutivos)
- **ROUGE-L**: subsecuencia común más larga (captura orden)

| Métrica | Este proyecto | Benchmark paper (CNN/DailyMail) |
|---|---|---|
| ROUGE-1 | calculado en app | 42.95 |
| ROUGE-2 | calculado en app | 20.82 |
| ROUGE-L | calculado en app | 30.62 |

---

## Visualización de atención

`attention_viz.py` extrae los pesos reales del modelo con `output_attentions=True` y genera:

- **8 subplots** de encoder self-attention (una por cabeza) — captura patrones distintos: sintaxis, correferencias, relaciones temáticas
- **Mapa promedio** de atención del encoder (todas las cabezas) — muestra dependencias globales
- **Cross-attention** decoder→encoder — qué tokens del artículo consulta el decoder al generar cada token del resumen

---

## Stack tecnológico

| Librería | Versión |
|---|---|
| Python | 3.10+ |
| PyTorch | ≥ 2.1.0 |
| HuggingFace Transformers | 4.40.0 |
| Streamlit | ≥ 1.33.0 |
| rouge-score | ≥ 0.1.2 |
| matplotlib | ≥ 3.8.0 |

El modelo se descarga automáticamente de Hugging Face en el primer uso (~1.6 GB en caché).

---

## Secciones de la interfaz Streamlit

1. **Contexto** — Marco del proyecto, problematica y solución
2. **Sumarización** — Entrada de texto libre, generación del resumen y métricas de compresión
3. **Atención** — Visualización interactiva de los 3 tipos de atención
4. **Evaluación ROUGE** — Benchmark sobre 3 pares artículo/referencia
5. **Arquitectura** — Dimensiones y parámetros del modelo

---

## Referencia

```
Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2019).
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.
arXiv:1910.13461
```

Modelo en Hugging Face: [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn)
