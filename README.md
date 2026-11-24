# Fake News Checker

Paste a headline, a link, or a social media handle. The app returns one of three verdicts:

| Verdict | Meaning |
|---|---|
| **Verified fake** | A fact-checker rates it false, Google Safe Browsing flags the link, or the domain is on your unreliable list |
| **Authenticated** | Fact-checkers rate it true, or a trusted outlet plus independent coverage support it with no red flags |
| **Unverified** | Anything else. This is the honest default, not an error |

The app is deliberately conservative. It would rather say "unverified" than guess.

## Why Python
Python has the best libraries for text and machine learning (so a classifier can be added later),
FastAPI makes a small async API easy (all checks run in parallel), and one codebase serves a
web app now. The same API can later back a mobile app or a desktop wrapper.

## Free data sources
| Source | Key needed | Used for |
|---|---|---|
| Google Fact Check Tools API | Free key | Published fact-checks of the claim |
| Google Safe Browsing | Free key (non-commercial) | Malicious or phishing links |
| GDELT DOC 2.0 | None | Whether trusted outlets cover the same story |
| RDAP (rdap.org) | None | Domain age |
| Wayback Machine availability API | None | Archive history |
| Bluesky public API | None | Account age and activity (other platforms have no free API) |

Without keys the app still runs. It just skips those checks and says so.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env   # then add your free keys, and export them
export FACTCHECK_API_KEY=... SAFE_BROWSING_API_KEY=...
python -m uvicorn app.main:app --reload
```
Open http://localhost:8000. Run the tests with `python -m pytest`.

## Trained text classifiers
Two models read the headline and vote. Their opinion is a weak nudge (max 15 points) and can never
decide a verdict alone, because a model that reads writing style cannot know whether a claim is true.

| Model | Looks at |
|---|---|
| Word TF-IDF + logistic regression | Which words and word pairs appear |
| Character TF-IDF + calibrated linear SVM | How it is written: capitals, punctuation, word shapes |

**No trained model ships with the repo.** You need real labelled data, which I could not
download from the build sandbox. Get a dataset and train:

```bash
# Short claims, closest to headlines (recommended as the main training set):
#   LIAR (Wang, 2017) https://doi.org/10.18653/v1/P17-2067
# Full articles from a different source, used ONLY to test generalisation:
#   ISOT (Ahmed, Traore and Saad, 2017) https://doi.org/10.1007/978-3-319-69155-8_9
python -m app.ml.train --data liar:data/liar/train.tsv --extra isot:data/isot
```
Training prints held-out metrics and writes `app/ml/artifacts/model.joblib` plus `model_card.json`.
The `--extra` score is the one to trust: it comes from a news source the models never saw.
If it is far below the in-source score, the models learned the source, not fake news.

### Measured results (first real training run)
Trained on LIAR train and validation. Threshold chosen on validation data only.

| Test set | Accuracy | Balanced accuracy | ROC AUC |
|---|---|---|---|
| LIAR, held-out split (same source) | 62% | 62% | 0.67 |
| LIAR, official test file | 64% | 64% | 0.68 |
| ISOT (different source, never trained on) | 52% | 54% | 0.59 |

Reading this honestly: on political claims the models are modestly better than a coin flip, and on
a different source they are close to useless. The app therefore caps the classifier's influence at
4 points when its outside-source AUC is below 0.65. Style models are a weak hint, and the fact-check,
blocklist and corroboration checks matter far more. Datasets are licensed for research use, so
`data/` is git-ignored. Re-download them rather than committing them.

Notes:
- The loaders strip source giveaways such as "(Reuters)" and links, which otherwise let a model cheat.
- `half-true` LIAR rows are dropped; `pants-fire`, `false` and `barely-true` count as fake.
- The model file is a pickle. Load only files you trained yourself. A scikit-learn version mismatch
  disables the model with a message instead of crashing.
- Without a model the app runs as before and lists the classifier under "Not checked".

## Customising
Edit `app/data/sources.json` to change the trusted, satire and unreliable domain lists.

## Limits
- Headlines are matched to fact-checks by text search, so paraphrased claims can be missed.
- Brand-new stories have no fact-checks yet and will often be "unverified".
- Heuristics (clickbait wording, odd domains) only nudge the result and never decide it alone.
- Trusted-outlet lists are a judgement call. Review them for your audience.
