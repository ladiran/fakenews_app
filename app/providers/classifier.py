"""Turns the trained models' output into evidence. Deliberately weak: a model that reads
writing style cannot know whether a claim is true, so it can nudge but never decide."""
import asyncio

from ..ml import data, model
from ..models import Evidence

MAX_WEIGHT = 15
WEAK_AUC = 0.65  # below this on an outside source, the model barely counts


async def check(client, text: str) -> list[Evidence]:  # client unused, kept for a uniform provider signature
    text = data.clean_text(text)
    if len(text) < 15:
        return []
    r = await asyncio.to_thread(model.score, text)  # sklearn is CPU-bound, keep it off the event loop
    p, t, per = r["p_fake"], r["threshold"], r["per_model"]
    agree_fake = all(v >= t for v in per.values())
    agree_real = all(v < 1 - t for v in per.values())
    ens = r["card"]["metrics"]["test"]["ensemble"]
    ext = r["card"]["metrics"].get("external") or {}
    worst_auc = min((m.get("roc_auc", 0.5) for m in ext.values()), default=None)
    cap = MAX_WEIGHT
    if worst_auc is None:
        caveat = " It was never tested on a different news source, so treat it as rough."
        cap = 8
    elif worst_auc < WEAK_AUC:
        caveat = (f" It scored poorly on a different news source (AUC {worst_auc:.2f}), "
                  "so its opinion is barely counted.")
        cap = 4
    else:
        caveat = ""
    detail = (f"Writing-style models estimate a {p:.0%} chance this reads like known fake news "
              f"(held-out accuracy {ens['balanced_accuracy']:.0%}).{caveat} It judges wording, not facts.")
    if agree_fake:
        return [Evidence(source="classifier", signal="style_fake", direction="fake",
                         weight=cap if p >= 0.8 else cap // 2, detail=detail)]
    if agree_real:
        return [Evidence(source="classifier", signal="style_real", direction="authentic",
                         weight=min(3, cap), detail=detail)]
    return [Evidence(source="classifier", signal="style_mixed", direction="neutral", detail=detail)]
