"""
inference.py
------------
Lógica de sumarización con BART.

Proceso completo:
    Texto (str)
        → Tokenizer BPE       → input_ids  [batch, seq_len]
        → Encoder             → hidden_states [batch, seq_len, d_model]
        → Decoder (beam=4)    → summary_ids  [batch, summary_len]
        → Tokenizer.decode    → resumen (str)
"""

import torch
from transformers import BartForConditionalGeneration, BartTokenizer


def tokenize(text: str, tokenizer: BartTokenizer, device: str, max_length: int = 1024):
    """
    Convierte texto a tensores usando BPE (Byte-Pair Encoding).

    BPE divide palabras desconocidas en subpalabras conocidas:
        "deforestation" → ["de", "forest", "ation"]
    Esto permite manejar vocabulario abierto con vocab fijo de 50.265 tokens.

    Tokens especiales que agrega BART automáticamente:
        <s>  (BOS, ID=0)  — inicio de secuencia
        </s> (EOS, ID=2)  — fin de secuencia

    Returns:
        dict con 'input_ids' y 'attention_mask' como tensores en `device`
    """
    return tokenizer(
        text,
        max_length=max_length,
        truncation=True,        # BART-large acepta máx 1024 tokens
        return_tensors="pt"
    ).to(device)


def summarize(
    text: str,
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    device: str,
    max_length: int = 130,
    min_length: int = 30,
    num_beams: int = 4,
    length_penalty: float = 2.0,
    no_repeat_ngram_size: int = 3,
) -> dict:
    """
    Genera un resumen abstractivo con beam search.

    Proceso interno paso a paso:
    1. Tokenización → input_ids
    2. Encoder bidireccional procesa TODOS los tokens simultáneamente
       (self-attention sin máscara causal)
    3. Decoder autoregresivo genera token a token:
       - Masked self-attention: solo ve tokens ya generados
       - Cross-attention: Q del decoder, K y V del encoder
         → así el resumen "mira" el artículo original en cada paso
    4. Beam search mantiene las 4 mejores hipótesis en paralelo
       (evita la solución greedy subóptima)
    5. Decodificación de IDs → texto

    Args:
        text          : artículo a resumir (inglés)
        max_length    : longitud máxima del resumen en tokens
        min_length    : longitud mínima del resumen en tokens
        num_beams     : número de beams para beam search
        length_penalty: > 1.0 favorece resúmenes más largos
        no_repeat_ngram_size: penaliza repetición de n-gramas

    Returns:
        dict con 'summary', 'input_tokens', 'output_tokens', 'compression_pct'
    """
    inputs = tokenize(text, tokenizer, device)
    input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():   # sin gradientes — solo inferencia
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    output_tokens = summary_ids.shape[1]

    n_orig = len(text.split())
    n_res  = len(summary.split())
    compression = round(100 * (1 - n_res / n_orig)) if n_orig > 0 else 0

    return {
        "summary"         : summary,
        "input_tokens"    : input_tokens,
        "output_tokens"   : output_tokens,
        "input_words"     : n_orig,
        "output_words"    : n_res,
        "compression_pct" : compression,
    }


def summarize_batch(
    texts: list[str],
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    device: str,
    max_length: int = 80,
    min_length: int = 20,
) -> list[dict]:
    """
    Sumariza una lista de textos usando el pipeline de alto nivel de HuggingFace.
    Más eficiente para múltiples textos seguidos.

    Returns:
        Lista de dicts con los mismos campos que `summarize()`
    """
    from transformers import pipeline

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )

    results = summarizer(texts, max_length=max_length, min_length=min_length, do_sample=False)

    output = []
    for text, result in zip(texts, results):
        summary = result["summary_text"]
        n_orig  = len(text.split())
        n_res   = len(summary.split())
        output.append({
            "summary"         : summary,
            "input_words"     : n_orig,
            "output_words"    : n_res,
            "compression_pct" : round(100 * (1 - n_res / n_orig)) if n_orig > 0 else 0,
        })

    return output