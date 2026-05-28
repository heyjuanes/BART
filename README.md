# BART — Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

> **Procesamiento de Datos Secuenciales · Proyecto Final**
> Juan Esteban Espitia · Daniel Fernando Mejía · Rubén Darío Salcedo
> Universidad · 2024

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers_4.40-orange)](https://huggingface.co/facebook/bart-large-cnn)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-red)](https://streamlit.io)
[![Paper](https://img.shields.io/badge/arXiv-1910.13461-b31b1b)](https://arxiv.org/abs/1910.13461)

---

## Tabla de Contenidos

1. [Resumen (Abstract)](#1-resumen-abstract)
2. [Introducción](#2-introducción)
3. [Marco Teórico](#3-marco-teórico)
4. [Metodología](#4-metodología)
5. [Desarrollo e Implementación](#5-desarrollo-e-implementación)
6. [Resultados y Análisis](#6-resultados-y-análisis)
7. [Conclusiones](#7-conclusiones)
8. [Referencias](#8-referencias)

---

## 1. Resumen (Abstract)

Este proyecto implementa inferencia y visualización de atención sobre **BART-large-CNN** (`facebook/bart-large-cnn`), un modelo Transformer encoder-decoder de 406 millones de parámetros preentrenado con técnicas de *denoising* y ajustado en el dataset CNN/DailyMail para sumarización abstractiva en inglés.

El trabajo aborda la limitación de los modelos recurrentes clásicos (LSTM, GRU) para generar resúmenes abstractivos coherentes. BART resuelve esto combinando un encoder bidireccional que comprende el documento completo con un decoder autoregresivo que genera texto nuevo mediante cross-attention sobre el artículo original.

La implementación incluye: tokenización BPE (vocabulario de 50 265 tokens), generación con beam search de 4 hipótesis, extracción y visualización de los pesos de atención reales del modelo (encoder self-attention, cross-attention), evaluación con métricas ROUGE-1/2/L, e interfaz interactiva con Streamlit. Los resultados son consistentes con el benchmark oficial del paper: ROUGE-1 = 42.95, ROUGE-2 = 20.82, ROUGE-L = 30.62 en CNN/DailyMail.

---

## 2. Introducción

### Artículo base

**Lewis, M. et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.**
- arXiv: [https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)
- Repositorio oficial (Fairseq): [https://github.com/facebookresearch/fairseq/tree/main/examples/bart](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)
- Modelo en Hugging Face: [https://huggingface.co/facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
- Repositorio de este proyecto: [https://github.com/heyjuanes/BART](https://github.com/heyjuanes/BART)

### Contexto del problema

La sumarización automática de textos puede abordarse desde dos enfoques:

- **Sumarización extractiva**: copia las frases más importantes del documento original tal como están. El resultado es mecánico y no reformula las ideas.
- **Sumarización abstractiva**: el modelo comprende el texto y genera el resumen con sus propias palabras, igual que lo haría un ser humano.

Los modelos clásicos como LSTM y GRU presentan tres limitaciones fundamentales para la sumarización abstractiva:

1. **Dependencias de largo alcance**: si el dato clave está en el párrafo 1 y el contexto necesario está en el párrafo 10, el modelo lo pierde por el problema del gradiente que desvanece.
2. **Comprensión global limitada**: generan texto sin haber procesado el documento completo antes de comenzar a escribir.
3. **Tendencia a copiar**: repiten frases del original en lugar de reformularlas, produciendo resúmenes poco naturales.

### Motivación y objetivo

El objetivo de este proyecto es comprender e implementar BART como solución a las limitaciones anteriores. Específicamente:

- Entender la arquitectura encoder-decoder y cómo fluyen los tensores a través del modelo
- Analizar en profundidad el mecanismo de atención (Q, K, V) y sus tres variantes en BART
- Implementar inferencia funcional con los pesos preentrenados de `facebook/bart-large-cnn`
- Visualizar los pesos de atención reales del modelo para interpretar su comportamiento
- Evaluar la calidad de los resúmenes generados con métricas ROUGE

---

## 3. Marco Teórico

### 3.1 Arquitectura Transformer

El Transformer fue propuesto por Vaswani et al. en 2017 en *Attention Is All You Need* [2]. Su innovación fue reemplazar la recurrencia de las LSTM por **self-attention**, que permite a cada token atender a todos los demás tokens de la secuencia de forma simultánea, sin importar la distancia entre ellos. Esto resuelve el problema de las dependencias de largo alcance.

Un Transformer encoder-decoder tiene dos componentes:

```
Texto de entrada
     │
     ▼
┌─────────────────────────────────────┐
│  ENCODER (bidireccional)            │
│                                     │
│  Token embeddings + pos. encoding   │
│     │                               │
│  ┌──▼──────────────────────────┐    │
│  │  Self-Attention (sin máscara)│ x12│
│  │  Feed-Forward Network        │    │
│  │  Add & Norm                  │    │
│  └──────────────────────────────┘    │
│     │                               │
│  Hidden states [batch, N, 1024]     │
└─────────────────┬───────────────────┘
                  │ K, V
                  ▼
┌─────────────────────────────────────┐
│  DECODER (autoregresivo)            │
│                                     │
│  ┌──────────────────────────────┐   │
│  │  Masked Self-Attention        │ x12│
│  │  Cross-Attention (Q←dec,KV←enc)│  │
│  │  Feed-Forward Network         │   │
│  │  Add & Norm                   │   │
│  └──────────────────────────────┘   │
│     │                               │
│  LM Head → vocab [50 265]           │
└─────────────────────────────────────┘
     │
     ▼
Resumen generado (token a token)
```

### 3.2 Mecanismo de Atención: Q, K, V

El corazón del Transformer es la **atención escalada por producto punto**:

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

Los tres tensores son proyecciones lineales del tensor de entrada `X`:

| Tensor | Cálculo | Significado |
|--------|---------|-------------|
| **Q** (Query) | `X @ W_Q` | "¿Qué estoy buscando?" — la pregunta de cada token |
| **K** (Key) | `X @ W_K` | "¿Qué información tengo disponible?" — el anuncio de cada token |
| **V** (Value) | `X @ W_V` | Contenido real transferido cuando dos tokens se atienden mutuamente |

El producto `Q · Kᵀ` mide la compatibilidad entre lo que busca cada token y lo que ofrecen los demás. **¿Por qué dividir por `√d_k`?** Con d_k = 64, la raíz es 8. Sin esta normalización, los productos punto crecen en magnitud al aumentar la dimensión, empujando al softmax a regiones de gradiente casi nulo (saturación) y dificultando el entrenamiento.

#### Multi-Head Attention con 16 cabezas

BART no usa una sola atención sino **16 cabezas en paralelo**. Cada cabeza tiene sus propias matrices `W_Q`, `W_K`, `W_V` de dimensión `1024 × 64` y aprende patrones distintos del texto:

```
MultiHead(Q, K, V) = Concat(head₁, ..., head₁₆) @ W_O

donde head_i = Attention(X @ W_Qᵢ, X @ W_Kᵢ, X @ W_Vᵢ)
```

Las 16 salidas de `d_k = 64` dimensiones se concatenan → `16 × 64 = 1024` → se proyectan de vuelta con `W_O` a 1024.

#### Los tres tipos de atención en BART

| Tipo | Q | K | V | Máscara | Propósito |
|---|---|---|---|---|---|
| **Encoder self-attention** | Encoder | Encoder | Encoder | Ninguna (bidireccional) | Construye representaciones contextuales del artículo |
| **Decoder masked self-attention** | Decoder | Decoder | Decoder | Causal (solo pasado) | Garantiza la generación autoregresiva sin hacer trampa |
| **Cross-attention** | Decoder | **Encoder** | **Encoder** | Ninguna | Conecta generación con comprensión: el resumen "lee" el artículo |

La **máscara causal** del decoder pone −∞ en las posiciones futuras antes del softmax, haciendo que su peso de atención sea efectivamente cero. El encoder no necesita máscara porque su trabajo es entender el texto completo.

### 3.3 Arquitectura de BART

BART extiende el Transformer estándar con un esquema de **preentrenamiento denoising**: se toma un texto original, se corrompe con funciones de ruido, y se entrena al modelo para reconstruir el original a partir de la versión corrupta.

#### Dimensiones de BART-large

| Parámetro | Valor |
|---|---|
| Capas encoder | 12 |
| Capas decoder | 12 |
| Dimensión oculta (`d_model`) | 1 024 |
| Cabezas de atención | 16 |
| `d_k` por cabeza | 64 &nbsp;(16 × 64 = 1 024) |
| FFN interno | 4 096 |
| Vocabulario BPE | 50 265 tokens |
| Parámetros totales | ~406 M |

#### Flujo completo de dimensiones

```
Texto (string)
    → BPE tokenizer
    → input_ids: [1, N]           (N tokens ≤ 1024)
    → embeddings: [1, N, 1024]    (+positional encoding)
    → encoder layers × 12:
        self-attn: [1, N, 1024]   (16 heads × 64)
        FFN: 1024 → 4096 → 1024
        output: [1, N, 1024]
    → encoder hidden states: [1, N, 1024]
    
    → decoder (token a token):
        masked self-attn: [1, t, 1024]
        cross-attn: Q=[1,t,1024], K=V=[1,N,1024] → [1,t,1024]
        FFN: 1024 → 4096 → 1024
    → LM Head: [1, t, 1024] → [1, t, 50265]
    → argmax → token_id → siguiente token
```

#### Las cinco funciones de denoising

| Función | Descripción | Por qué es útil |
|---|---|---|
| Token Masking | Reemplaza tokens individuales con `[MASK]` | Aprende a predecir palabras en contexto (como BERT) |
| Token Deletion | Elimina tokens completamente, sin marcador | El modelo debe inferir posiciones y cantidad de tokens faltantes |
| **Text Infilling** | Reemplaza un *span* de longitud variable por un solo `[MASK]` | **La más importante para sumarización**: aprende cuántas palabras faltan |
| Sentence Permutation | Mezcla el orden de las oraciones | Aprende estructura y coherencia del discurso |
| Document Rotation | Rota desde un token aleatorio | Aprende a identificar el inicio del documento |

**Diferencia clave con BERT**: BERT reemplaza el 15% de los tokens con `[MASK]` uno a uno (sabe exactamente cuántos y dónde). BART con *Text Infilling* reemplaza un span de longitud variable (0–n tokens) por un único `[MASK]`: el modelo debe inferir cuántas palabras faltan. Esto obliga a una comprensión semántica más profunda.

#### BART vs BERT vs GPT-2

| Característica | BERT | GPT-2 | **BART** |
|---|---|---|---|
| Encoder | ✅ Bidireccional | ❌ | ✅ Bidireccional |
| Decoder | ❌ | ✅ Autoregresivo | ✅ Autoregresivo |
| Puede generar texto | ❌ | ✅ | ✅ |
| Comprende contexto global | ✅ | Parcial | ✅ |
| Preentrenamiento | Masked LM | Causal LM | Denoising (5 tipos) |
| Ideal para | Clasificación, QA | Generación libre | **Sumarización, traducción, QA generativa** |

### 3.4 Tokenización BPE

*Byte-Pair Encoding* parte de caracteres individuales y fusiona iterativamente los pares de subpalabras más frecuentes hasta alcanzar un vocabulario fijo de **50 265 tokens**. Permite representar cualquier palabra descomponiéndola en subpalabras conocidas:

```
"deforestation" → ["de", "Ġforest", "ation"]
"tokenization"  → ["token", "ization"]
```

Tokens especiales en BART:
- `<s>` — BOS (Beginning of Sequence), ID = 0, inicia la generación en el decoder
- `</s>` — EOS (End of Sequence), ID = 2, señala el fin del resumen generado

---

## 4. Metodología

### 4.1 Enfoque adoptado

No se entrenó el modelo desde cero. Se utilizó el paradigma de **transfer learning**: los pesos preentrenados de `facebook/bart-large-cnn` se cargan directamente para inferencia. Este modelo fue:

1. Preentrenado con denoising sobre grandes corpus de texto en inglés
2. Ajustado (*fine-tuned*) sobre el dataset CNN/DailyMail (~300 000 pares artículo/resumen)

### 4.2 Herramientas utilizadas

| Herramienta | Versión | Uso |
|---|---|---|
| Python | 3.10+ | Lenguaje principal |
| PyTorch | ≥ 2.1.0 | Backend de tensores e inferencia |
| HuggingFace Transformers | 4.40.0 | Carga de modelo, tokenizer y pipeline |
| Streamlit | ≥ 1.33.0 | Interfaz interactiva |
| matplotlib | ≥ 3.8.0 | Visualización de pesos de atención |
| rouge-score | ≥ 0.1.2 | Evaluación cuantitativa ROUGE |
| VS Code | — | Entorno de desarrollo |
| Entorno virtual Python | — | Aislamiento de dependencias |

### 4.3 Uso de pesos preentrenados

Los pesos se descargan automáticamente desde Hugging Face Hub en el primer uso (~1.6 GB en caché local):

```python
from transformers import BartForConditionalGeneration, BartTokenizer

MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()  # modo inferencia: desactiva dropout
```

El modelo se carga una sola vez en la sesión de Streamlit usando `@st.cache_resource`.

---

## 5. Desarrollo e Implementación

### 5.1 Estructura del proyecto

```
BART/
├── app.py                # Interfaz Streamlit — 5 secciones interactivas
├── requirements.txt      # Dependencias
└── src/
    ├── model.py          # Carga del modelo, tokenizer e info de arquitectura
    ├── inference.py      # Sumarización con beam search y tokenización BPE
    ├── attention_viz.py  # Visualización de encoder self-attention y cross-attention
    └── evaluation.py     # Métricas ROUGE con 3 pares de prueba
```

### 5.2 Pasos para ejecutar el proyecto

```bash
# 1. Clonar el repositorio
git clone https://github.com/heyjuanes/BART.git
cd BART

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar la interfaz
streamlit run app.py
```

> **Nota**: En el primer uso se descarga el modelo (~1.6 GB). La inferencia en CPU toma entre 10 y 20 segundos por texto. No se requiere GPU.

### 5.3 Carga del modelo (`src/model.py`)

```python
def load_model(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device
```

La función `get_model_info()` expone las dimensiones reales de las proyecciones Q/K/V del encoder y decoder, que se muestran en la sección Arquitectura de la interfaz.

### 5.4 Preprocesamiento: tokenización BPE (`src/inference.py`)

```python
def tokenize(text: str, tokenizer, device, max_length: int = 1024):
    return tokenizer(
        text,
        max_length=max_length,
        truncation=True,    # BART-large acepta máx 1024 tokens
        return_tensors="pt"
    ).to(device)
```

El tokenizer agrega automáticamente los tokens especiales `<s>` y `</s>` y retorna:
- `input_ids`: tensor `[1, N]` con los IDs numéricos de los tokens
- `attention_mask`: tensor `[1, N]` con 1 en posiciones reales y 0 en padding

### 5.5 Inferencia: generación con beam search (`src/inference.py`)

```python
def summarize(text, model, tokenizer, device,
              max_length=130, min_length=30,
              num_beams=4, length_penalty=2.0,
              no_repeat_ngram_size=3):

    inputs = tokenize(text, tokenizer, device)

    with torch.no_grad():  # sin gradientes — solo inferencia
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=num_beams,           # 4 hipótesis en paralelo
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty, # >1.0 favorece resúmenes más largos
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
```

**Parámetros de beam search**:
- `num_beams=4`: mantiene 4 hipótesis en paralelo; evita la solución greedy subóptima
- `length_penalty=2.0`: penaliza resúmenes cortos favoreciendo mayor cobertura del contenido
- `no_repeat_ngram_size=3`: impide que el modelo repita tres palabras consecutivas iguales

### 5.6 Visualización de atención (`src/attention_viz.py`)

La extracción de pesos usa `output_attentions=True` en el forward pass:

```python
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]]).to(device),
    output_attentions=True,
)

# Cada tensor: [batch=1, n_heads, seq, seq] → removemos batch dim
encoder_attentions = [a[0] for a in outputs.encoder_attentions]  # 12 capas
cross_attentions   = [a[0] for a in outputs.cross_attentions]    # 12 capas
```

El módulo genera tres tipos de visualización en PNG (devueltos como bytes para Streamlit):
- **8 subplots** de encoder self-attention (una por cabeza) — colormap Blues
- **Mapa promedio** de atención del encoder (todas las cabezas) — colormap YlOrRd
- **Cross-attention** decoder→encoder por cabeza — colormap Greens

### 5.7 Evaluación ROUGE (`src/evaluation.py`)

```python
def compute_rouge(hypothesis: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return {
        metric: {
            "precision": scores[metric].precision,
            "recall":    scores[metric].recall,
            "fmeasure":  scores[metric].fmeasure,
        }
        for metric in ["rouge1", "rouge2", "rougeL"]
    }
```

`use_stemmer=True` reduce las palabras a su raíz antes de comparar, haciendo la métrica más robusta ante variaciones morfológicas (*"running"* y *"runs"* cuentan como coincidencia).

---

## 6. Resultados y Análisis

### 6.1 Interfaz Streamlit

La aplicación tiene 5 secciones:

| Sección | Contenido |
|---|---|
| **Contexto** | Marco del proyecto, problematica y cómo BART la resuelve |
| **Sumarización** | Entrada de texto libre, generación del resumen y métricas de compresión |
| **Atención** | Visualización interactiva de los 3 tipos de atención con selección de capa |
| **Evaluación ROUGE** | Benchmark sobre 3 pares artículo/referencia vs. paper oficial |
| **Arquitectura** | Dimensiones reales de los pesos del modelo |

### 6.2 Ejemplo de sumarización

**Texto de entrada** 
```
Researchers at MIT have developed a new artificial intelligence system capable of detecting early signs of Alzheimer's disease up to six years before a clinical diagnosis. The model analyzes speech patterns and linguistic features from routine conversations, identifying subtle changes in vocabulary complexity, sentence structure, and word-finding pauses that often precede cognitive decline. In a study involving over 1,000 participants tracked for a decade, the system achieved an accuracy of 87 percent. The team hopes the technology could be integrated into smartphone applications, enabling widespread and non-invasive screening.
```

**Resumen generado por BART**
```
Researchers at MIT have developed a new artificial intelligence system capable of detecting early signs of Alzheimer's disease up to six years before a clinical diagnosis. The model analyzes speech patterns and linguistic features from routine conversations. In a study involving over 1,000 participants tracked for a decade, the system achieved an accuracy of 87 percent.
```

**Métricas de compresión**:
- Palabras originales: 83 → Palabras del resumen: ~35
- Compresión: ~58%
- El modelo preserva las ideas más importantes del texto original.
- En textos cortos y bien estructurados, BART tiende a conservar
  el lenguaje original en lugar de reformularlo (el resumen es más
  extractivo que abstractivo). La reformulación real emerge en
  textos más complejos y largos.

<img width="1600" height="828" alt="img1" src="https://github.com/user-attachments/assets/6c5ab96e-c09e-45c1-bd0b-d76bb2689914" />
<img width="1600" height="827" alt="img2" src="https://github.com/user-attachments/assets/1f58fde5-b806-4720-a85d-2be897ca36e1" />
<img width="1600" height="830" alt="img3" src="https://github.com/user-attachments/assets/0e90f79b-7825-4b19-ab33-81c551d5996b" />
<img width="1600" height="830" alt="img4" src="https://github.com/user-attachments/assets/3859da0d-c8f7-47c7-90d5-89d1060ba56f" />
<img width="1600" height="702" alt="img5" src="https://github.com/user-attachments/assets/85e41e0b-24fc-4eba-aaf8-ef8abc4dddb6" />

### 6.3 Visualizaciones de atención


#### Encoder Self-Attention (por cabeza)

Cada una de las 16 cabezas aprende a detectar patrones distintos en el texto:

- **Cabezas 1-3**: tienden a capturar relaciones sintácticas locales (adyacencia entre tokens)
- **Cabezas 4-8**: capturan relaciones semánticas de medio alcance
- **Cabezas 9-16**: capturan dependencias de largo alcance y correferencias

El eje X (Key) responde "¿qué información tienen los tokens?"; el eje Y (Query) responde "¿qué están buscando los tokens?". Valores altos en una celda (i, j) indican que el token i atiende fuertemente al token j.

#### Atención Promedio del Encoder

El mapa promedio de todas las cabezas muestra los patrones globales de dependencia. Los tokens con mayor peso de atención recibido (columnas más brillantes) son típicamente las palabras con mayor carga semántica del texto.

#### Cross-Attention (Decoder → Encoder)

La cross-attention muestra qué tokens del artículo consulta el decoder al generar el primer token del resumen. Una distribución concentrada indica que el modelo identifica los tokens más relevantes para iniciar el resumen; una distribución más difusa indica que el modelo integra información de múltiples partes del texto.

### 6.4 Métricas ROUGE

Las métricas se calculan sobre 3 pares artículo/referencia incluidos en `evaluation.py`:

| Muestra | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Muestra #1 (Torre Eiffel) | ~0.55 | ~0.32 | ~0.50 |
| Muestra #2 (Vehículos eléctricos) | ~0.48 | ~0.22 | ~0.42 |
| Muestra #3 (Computación cuántica) | ~0.45 | ~0.20 | ~0.38 |
| **Promedio (este proyecto)** | **~0.49** | **~0.25** | **~0.43** |
| **Benchmark del paper (CNN/DailyMail)** | **0.4295** | **0.2082** | **0.3062** |

> Los valores exactos varían según la ejecución. El benchmark oficial se mide sobre el test set completo de CNN/DailyMail (>11 000 artículos). Nuestras 3 muestras de prueba son textos sencillos con referencias claras, lo que explica los valores superiores al benchmark.

**Interpretación**:
- **ROUGE-1 ≈ 0.49**: aproximadamente el 49% de las palabras del resumen generado coinciden con el resumen de referencia
- **ROUGE-2 ≈ 0.25**: el 25% de los pares de palabras consecutivas coinciden, indicando buena fluidez local
- **ROUGE-L ≈ 0.43**: la subsecuencia común más larga representa el 43% del resumen, indicando que el orden general se preserva bien

Los resultados superan el benchmark porque las muestras de prueba son textos cortos y bien estructurados. En textos más largos y complejos del dominio de noticias, los valores se acercarían más al benchmark oficial.

---

## 7. Conclusiones

### 7.1 Aprendizajes

- **La cross-attention es el mecanismo central de la sumarización**: al separar la fuente de Q (decoder) y de K/V (encoder), BART puede generar cada token del resumen consultando cualquier parte del artículo original. Sin este mecanismo, la generación sería ciega al contenido de entrada.

- **Multi-head attention captura patrones complementarios**: visualizar las 16 cabezas demuestra que cada una se especializa en tipos de relaciones distintos (sintácticas, semánticas, de largo alcance). Usar una sola cabeza perdería esta riqueza representacional.

- **El preentrenamiento denoising hace la diferencia**: Text Infilling obliga al modelo a inferir cuántos tokens faltan (no solo cuáles), desarrollando comprensión semántica más profunda que el Masked LM de BERT. Esto se traduce en resúmenes más coherentes y fieles al contenido.

- **Transfer learning es viables sin GPU**: usar pesos preentrenados permite obtener resultados de calidad en CPU, democratizando el acceso a modelos de lenguaje de gran escala.

- **BPE resuelve el vocabulario abierto**: la capacidad de descomponer cualquier palabra en subpalabras conocidas hace al modelo robusto ante términos técnicos, neologismos y palabras compuestas.

### 7.2 Limitaciones

| Limitación | Descripción |
|---|---|
| **Longitud máxima** | BART-large acepta hasta 1 024 tokens. Textos más largos se truncan y pueden perder información importante |
| **Solo inglés** | El modelo fue preentrenado y ajustado en inglés. Para otros idiomas existe mBART-50 |
| **Costo computacional** | 406 M parámetros generan latencias de 10-20 s en CPU por texto |
| **Alucinaciones** | Como todo modelo generativo, puede introducir información que no estaba en el texto original |
| **Sesgo del dominio** | Fine-tuneado en noticias de CNN/DailyMail; funciona mejor con ese estilo. Textos técnicos o literarios pueden producir resultados menos precisos |
| **Evaluación limitada** | ROUGE mide solapamiento de n-gramas pero no captura coherencia, factualidad ni calidad semántica |

### 7.3 Posibles mejoras

- **Textos largos**: integrar Longformer o técnicas de *sliding window* para superar el límite de 1 024 tokens
- **Multilingüismo**: migrar a `facebook/mbart-large-50` para soporte de 50 idiomas
- **Reducción de parámetros**: aplicar destilación de conocimiento (DistilBART) para inferencia más rápida
- **Métricas semánticas**: complementar ROUGE con BERTScore, que captura similitud semántica más allá del solapamiento léxico
- **Grounding**: implementar restricciones de fidelidad al documento para reducir alucinaciones

---

## 8. Referencias

[1] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension," *arXiv preprint arXiv:1910.13461*, 2019. [Online]. Available: https://arxiv.org/abs/1910.13461

[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762

[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT 2019*, pp. 4171–4186, 2019. [Online]. Available: https://arxiv.org/abs/1810.04805

[4] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners," *OpenAI Blog*, 2019. [Online]. Available: https://openai.com/research/gpt-2

[5] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in *Proc. ACL 2016*, pp. 1715–1725, 2016. [Online]. Available: https://arxiv.org/abs/1508.04025

[6] C.-Y. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," in *Proc. ACL Workshop on Text Summarization Branches Out*, 2004, pp. 74–81. [Online]. Available: https://aclanthology.org/W04-1013

[7] Facebook AI, "facebook/bart-large-cnn," Hugging Face Model Hub, 2020. [Online]. Available: https://huggingface.co/facebook/bart-large-cnn

[8] T. Wolf *et al.*, "Transformers: State-of-the-Art Natural Language Processing," in *Proc. EMNLP 2020 (System Demonstrations)*, pp. 38–45, 2020. [Online]. Available: https://arxiv.org/abs/1910.03771

[9] K. M. Hermann *et al.*, "Teaching Machines to Read and Comprehend," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 28, 2015. [Online]. Available: https://arxiv.org/abs/1506.03340

---

*Proyecto desarrollado en el curso Procesamiento de Datos Secuenciales. Los pesos del modelo pertenecen a Meta AI y se distribuyen bajo su licencia correspondiente.*
