"""Turns the trained models' output into evidence. Deliberately weak: a model that reads
writing style cannot know whether a claim is true, so it can nudge but never decide."""
import asyncio

from ..ml import data, model
from ..models import Evidence

MAX_WEIGHT = 15


async def check(client, text: str) -> list[Evidence]:  # client unused, kept for a uniform provider signature
    text = data.clean_text(text)
    if len(text) < 15:
        return []
    r = await asyncio.to_thread(model.score, text)  # sklearn is CPU-bound, keep it off the event loop
    p, t, per = r["p_fake"], r["threshold"], r["per_model"]
    agree_fake = all(v >= t for v in per.values())
    agree_real = all(v < 1 - t for v in per.values())
    f1 = r["card"]["metrics"]["test"]["ensemble"]["f1_fake"]
    ext = r["card"]["metrics"].get("external")
    caveat = ("" if ext else " (never tested on a different news source, so treat as rough)")
    detail = (f"Writing-style models estimate a {p:.0%} chance this reads like known fake news "
              f"(held-out F1 {f1:.2f}){caveat}. It judges wording, not facts.")
    if agree_fake:
        return [Evidence(source="classifier", signal="style_fake", direction="fake",
                         weight=MAX_WEIGHT if p >= 0.8 else 8, detail=detail)]
    if agree_real:
        return [Evidence(source="classifier", signal="style_real", direction="authentic",
                         weight=3, detail=detail)]
    return [Evidence(source="classifier", signal="style_mixed", direction="neutral", detail=detail)]
