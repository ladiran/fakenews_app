"""Plumbing tests only. The toy data below is SYNTHETIC and says nothing about real accuracy."""
import csv
import random

import pytest

from app.ml import data, model, train
from app.providers import classifier

random.seed(0)
_FAKE_A = ["SHOCKING", "Miracle", "Secret", "Banned", "Exposed", "Unbelievable", "Leaked"]
_FAKE_B = ["cure doctors hate", "truth the elites hide", "proof they lied", "plot to silence you", "trick melts fat"]
_REAL_A = ["Council approves", "Committee reviews", "Ministry publishes", "Board schedules", "Agency releases"]
_REAL_B = ["quarterly budget report", "road maintenance plan", "annual housing figures", "new library hours", "water testing results"]


def toy(n=150):
    X, y = [], []
    for i in range(n):
        X.append(f"{random.choice(_FAKE_A)}!! {random.choice(_FAKE_B)} number {i}")
        y.append(1)
        X.append(f"{random.choice(_REAL_A)} {random.choice(_REAL_B)} item {i}")
        y.append(0)
    return X, y


def write_csv(path, X, y):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(zip(X, y))


@pytest.fixture
def trained(tmp_path, monkeypatch):
    X, y = toy()
    write_csv(tmp_path / "a.csv", X, y)
    random.seed(1)
    X2, y2 = toy(60)
    write_csv(tmp_path / "ext.csv", X2, y2)
    out = tmp_path / "art"
    card = train.train([f"csv:{tmp_path/'a.csv'}"], [f"csv:{tmp_path/'ext.csv'}"], out)
    monkeypatch.setenv("FAKENEWS_MODEL_PATH", str(out / "model.joblib"))
    model._load.cache_clear()
    yield card
    model._load.cache_clear()


def test_clean_text_removes_source_tells():
    s = "WASHINGTON (Reuters) - The bill passed. Image via Getty Images https://t.co/x @bob"
    out = data.clean_text(s)
    assert "Reuters" not in out and "http" not in out and "@bob" not in out and "Getty" not in out
    assert "bill passed" in out

def test_dedupe_drops_dupes_and_short():
    X, y = data.dedupe(["Hello world this is long", "hello world this is long", "short"], [0, 0, 1])
    assert X == ["Hello world this is long"]

def test_liar_loader(tmp_path):
    rows = [["1.json", "pants-fire", "Statement one is long enough"], ["2.json", "true", "Statement two is long enough"],
            ["3.json", "half-true", "Statement three is long enough"], ["bad"]]
    with open(tmp_path / "t.tsv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(rows)
    X, y = data.load_liar(tmp_path / "t.tsv")
    assert y == [1, 0] and len(X) == 2  # half-true dropped, malformed row skipped

def test_isot_loader(tmp_path):
    for name, title in (("Fake.csv", "Fake title here"), ("True.csv", "True title here")):
        with open(tmp_path / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["title", "text"]); w.writeheader(); w.writerow({"title": title, "text": "body"})
    X, y = data.load_isot(tmp_path)
    assert sorted(y) == [0, 1]

def test_csv_rejects_bad_labels(tmp_path):
    write_csv(tmp_path / "b.csv", ["x" * 20], ["maybe"])
    with pytest.raises(ValueError):
        data.load_csv(tmp_path / "b.csv")

def test_bad_spec():
    with pytest.raises(ValueError):
        data.load("nonsense")

def test_train_produces_card_and_external_metrics(trained):
    assert trained["metrics"]["test"]["ensemble"]["accuracy"] > 0.9  # toy data is trivially separable
    assert any(k.startswith("csv:") for k in trained["metrics"]["external"])
    assert {"word_logreg", "char_svm"} <= set(trained["metrics"]["test"])

def test_score_separates_toy_classes(trained):
    assert model.score("SHOCKING!! Secret cure doctors hate number 3")["p_fake"] > 0.5
    assert model.score("Council approves quarterly budget report item 3")["p_fake"] < 0.5

@pytest.mark.asyncio
async def test_classifier_evidence_is_weak(trained):
    out = await classifier.check(None, "SHOCKING!! Banned trick melts fat number 9")
    assert out and not any(e.decisive for e in out) and out[0].weight <= classifier.MAX_WEIGHT

@pytest.mark.asyncio
async def test_missing_model_is_reported_not_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKENEWS_MODEL_PATH", str(tmp_path / "nope.joblib"))
    model._load.cache_clear()
    with pytest.raises(model.ModelUnavailable):
        await classifier.check(None, "Some headline that is long enough")
    model._load.cache_clear()

def test_too_few_examples_refuses(tmp_path):
    write_csv(tmp_path / "s.csv", [f"example text number {i} long" for i in range(10)], [i % 2 for i in range(10)])
    with pytest.raises(SystemExit):
        train.train([f"csv:{tmp_path/'s.csv'}"], [], tmp_path / "o")
