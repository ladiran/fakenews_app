"""Dataset loaders and text cleaning.

We do NOT bundle any dataset. Download one yourself (see README) and point the
trainer at it. Supported layouts:

* liar  : LIAR tab-separated files (train.tsv). Short political claims, six-way labels.
          Close to the length of a headline, so it suits this app best.
* isot  : ISOT-style folder with Fake.csv and True.csv (columns title, text).
* csv   : any CSV with a text column and a 0/1 label column (1 = fake).
"""
import csv
import re
from pathlib import Path

FAKE, REAL = 1, 0

# LIAR labels. "half-true" is dropped by default because it is neither side.
LIAR_MAP = {"pants-fire": FAKE, "false": FAKE, "barely-true": FAKE,
            "mostly-true": REAL, "true": REAL}

# Source "tells" that let a model cheat by recognising the outlet, not the content.
_LEAKS = [
    re.compile(r"^[A-Z][A-Z .,/'-]{2,40}\((Reuters|AP|AFP)\)\s*[-–]?\s*", re.M),
    re.compile(r"\((Reuters|AP|AFP)\)", re.I),
    re.compile(r"https?://\S+"),
    re.compile(r"\b(pic\.twitter\.com|twitter\.com)/\S+", re.I),
    re.compile(r"\b(featured image|image via|photo by|getty images?)\b[^.\n]*", re.I),
    re.compile(r"@\w+"),
]


def clean_text(s: str) -> str:
    for rx in _LEAKS:
        s = rx.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_liar(path: str | Path) -> tuple[list[str], list[int]]:
    X, y = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3:
                continue
            label, text = row[1].strip().lower(), row[2]
            if label in LIAR_MAP:
                X.append(clean_text(text))
                y.append(LIAR_MAP[label])
    return X, y


def load_isot(folder: str | Path, use_body: bool = False) -> tuple[list[str], list[int]]:
    folder = Path(folder)
    X, y = [], []
    for fname, label in (("Fake.csv", FAKE), ("True.csv", REAL)):
        with open(folder / fname, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                text = row.get("title", "")
                if use_body:
                    text += ". " + row.get("text", "")
                X.append(clean_text(text))
                y.append(label)
    return X, y


def load_csv(path: str | Path, text_col: str = "text", label_col: str = "label") -> tuple[list[str], list[int]]:
    X, y = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lab = str(row[label_col]).strip()
            if lab not in ("0", "1"):
                raise ValueError(f"label column must be 0 or 1, got {lab!r}")
            X.append(clean_text(row[text_col]))
            y.append(int(lab))
    return X, y


def load(spec: str, **kw) -> tuple[list[str], list[int]]:
    """spec looks like 'liar:path/train.tsv', 'isot:path/folder' or 'csv:path/file.csv'."""
    kind, _, path = spec.partition(":")
    loaders = {"liar": load_liar, "isot": load_isot, "csv": load_csv}
    if kind not in loaders or not path:
        raise ValueError("dataset spec must look like liar:PATH, isot:PATH or csv:PATH")
    return loaders[kind](path, **kw)


def dedupe(X: list[str], y: list[int], min_len: int = 15) -> tuple[list[str], list[int]]:
    """Drop short and duplicate texts. Duplicates across train and test inflate scores."""
    seen, ox, oy = set(), [], []
    for t, l in zip(X, y):
        key = t.lower()
        if len(t) < min_len or key in seen:
            continue
        seen.add(key)
        ox.append(t)
        oy.append(l)
    return ox, oy
