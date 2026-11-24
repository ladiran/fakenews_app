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

## Customising
Edit `app/data/sources.json` to change the trusted, satire and unreliable domain lists.

## Limits
- Headlines are matched to fact-checks by text search, so paraphrased claims can be missed.
- Brand-new stories have no fact-checks yet and will often be "unverified".
- Heuristics (clickbait wording, odd domains) only nudge the result and never decide it alone.
- Trusted-outlet lists are a judgement call. Review them for your audience.
