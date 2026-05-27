"""
evaluation.py
-------------
Evaluación de calidad de resúmenes con métricas ROUGE.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
  - ROUGE-1 : solapamiento de unigramas  (palabras individuales)
  - ROUGE-2 : solapamiento de bigramas   (pares de palabras consecutivas)
  - ROUGE-L : subsecuencia común más larga (captura orden)

Benchmark oficial de BART-large-CNN en CNN/DailyMail:
  ROUGE-1 = 42.95 | ROUGE-2 = 20.82 | ROUGE-L = 30.62
"""

import numpy as np
from rouge_score import rouge_scorer

# Pares artículo / resumen de referencia humana para evaluación
EVAL_PAIRS = [
    {
        "articulo": (
            "The Eiffel Tower located in Paris France was built between 1887 and 1889 "
            "as the entrance arch for the World Fair. Designed by engineer Gustave Eiffel "
            "the tower stands 330 meters tall and was the world tallest man-made structure "
            "for 41 years. Today it attracts nearly 7 million visitors annually making it "
            "one of the most visited monuments in the world."
        ),
        "referencia": (
            "The Eiffel Tower was built in Paris in 1889 for the World Fair. "
            "Designed by Gustave Eiffel it stands 330 meters and receives 7 million visitors per year."
        ),
    },
    {
        "articulo": (
            "Electric vehicles are gaining popularity worldwide as governments push for greener "
            "transportation. Tesla remains the market leader but traditional automakers like "
            "Volkswagen GM and Toyota are rapidly expanding their EV lineups. Battery technology "
            "continues to improve with newer models offering ranges exceeding 500 kilometers on "
            "a single charge. Charging infrastructure is also expanding though gaps remain in rural areas."
        ),
        "referencia": (
            "Electric vehicles are growing with Tesla leading the market. Traditional automakers "
            "are expanding EV lineups. Battery range now exceeds 500 km but charging gaps remain "
            "in rural areas."
        ),
    },
    {
        "articulo": (
            "Scientists have made a major breakthrough in quantum computing achieving a new record "
            "for qubit stability. Researchers at IBM managed to maintain coherence for over 10 minutes "
            "using a novel error correction technique. This development could accelerate the timeline "
            "for practical quantum computing applications in cryptography drug discovery and materials science."
        ),
        "referencia": (
            "IBM scientists set a qubit stability record of 10 minutes using a new error correction method. "
            "The breakthrough could speed up quantum computing applications in multiple fields."
        ),
    },
]

# Benchmark oficial del paper (CNN/DailyMail test set)
BENCHMARK = {
    "rouge1": 0.4295,
    "rouge2": 0.2082,
    "rougeL": 0.3062,
}


def compute_rouge(hypothesis: str, reference: str) -> dict:
    """
    Calcula ROUGE-1, ROUGE-2 y ROUGE-L entre un resumen generado y uno de referencia.

    Args:
        hypothesis : resumen generado por el modelo
        reference  : resumen de referencia humana

    Returns:
        dict con precision, recall y f-measure para cada métrica
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        metric: {
            "precision": scores[metric].precision,
            "recall"   : scores[metric].recall,
            "fmeasure" : scores[metric].fmeasure,
        }
        for metric in ["rouge1", "rouge2", "rougeL"]
    }


def evaluate_batch(summaries: list[str], pairs: list[dict] = None) -> dict:
    """
    Evalúa una lista de resúmenes generados contra referencias humanas.

    Args:
        summaries : resúmenes generados por BART (mismo orden que EVAL_PAIRS)
        pairs     : pares artículo/referencia (por defecto usa EVAL_PAIRS)

    Returns:
        dict con scores por muestra y promedios globales
    """
    if pairs is None:
        pairs = EVAL_PAIRS

    scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = []
    r1, r2, rL = [], [], []

    for summary, pair in zip(summaries, pairs):
        s = scorer.score(pair["referencia"], summary)
        results.append({
            "summary"  : summary,
            "rouge1"   : s["rouge1"].fmeasure,
            "rouge2"   : s["rouge2"].fmeasure,
            "rougeL"   : s["rougeL"].fmeasure,
        })
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)

    return {
        "per_sample" : results,
        "averages"   : {
            "rouge1": float(np.mean(r1)),
            "rouge2": float(np.mean(r2)),
            "rougeL": float(np.mean(rL)),
        },
        "benchmark"  : BENCHMARK,
    }


def format_rouge_table(eval_results: dict) -> str:
    """Retorna una tabla de texto formateada con los resultados ROUGE."""
    lines = [
        f"{'Muestra':<12} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}",
        "-" * 45,
    ]
    for i, s in enumerate(eval_results["per_sample"]):
        lines.append(f"  Muestra #{i+1}   {s['rouge1']:>10.3f} {s['rouge2']:>10.3f} {s['rougeL']:>10.3f}")

    avg = eval_results["averages"]
    bm  = eval_results["benchmark"]
    lines += [
        "-" * 45,
        f"  Promedio     {avg['rouge1']:>10.3f} {avg['rouge2']:>10.3f} {avg['rougeL']:>10.3f}",
        f"  Benchmark    {bm['rouge1']:>10.3f} {bm['rouge2']:>10.3f} {bm['rougeL']:>10.3f}",
    ]
    return "\n".join(lines)