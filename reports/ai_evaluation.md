# AI Evaluation Report

Generated: 2026-08-08T18:18:05+00:00
Model version: `1.0.0` · dataset seed: `42` · training time: 67.4s

This is the "AI testing evidence" deliverable from the project spec (§13). It reports
what the models get right, **what they get wrong**, and where they should not be trusted.

---

## 1. What the AI receives, does, and returns

| | |
|---|---|
| **Input** | The citizen's free-text complaint (plus location, which is not used by the model) |
| **Processing** | Normalise text → TF-IDF word (1–2gram) + character (3–5gram) vectors → linear classifiers; the priority model additionally receives 9 engineered severity features |
| **Output** | `category` (7 classes) + `priority` (4 classes), each with a calibrated confidence and runner-up candidates; an extractive one-line summary; a routed department; keywords |
| **Fallback** | If a model artifact is missing, a keyword-rule analyzer takes over so submissions never fail |

---

## 2. Dataset

Generated from composable templates (see `app/ml/dataset.py`) — issue phrasings ×
locations × durations × impact clauses × openers/closers, with 35% of samples degraded
by typos, casing changes, dropped punctuation and Hindi/English code-mixing to mirror
real submission channels.

- **Total samples:** 6,000 (deduplicated)
- **Train / test split:** 4,800 / 1,200 — stratified on category, seed `42`

**Category balance**

- `drainage`: 810
- `electricity`: 860
- `other`: 904
- `road`: 882
- `safety`: 709
- `waste`: 873
- `water`: 962

**Priority balance**

- `critical`: 1500
- `high`: 1500
- `low`: 1500
- `medium`: 1500

---

## 3. Category classifier

TF-IDF (word 1–2gram, weight 1.0 + char_wb 3–5gram, weight 0.6) → `LinearSVC`
wrapped in `CalibratedClassifierCV` so it emits usable probabilities.

- **Accuracy:** 1.000
- **Macro F1:** 1.000
- **Weighted F1:** 1.000
- **5-fold CV accuracy (train set):** 1.000

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| drainage | 1.000 | 1.000 | 1.000 | 162 |
| electricity | 1.000 | 1.000 | 1.000 | 172 |
| other | 1.000 | 1.000 | 1.000 | 181 |
| road | 1.000 | 1.000 | 1.000 | 176 |
| safety | 1.000 | 1.000 | 1.000 | 142 |
| waste | 1.000 | 1.000 | 1.000 | 175 |
| water | 1.000 | 1.000 | 1.000 | 192 |

**Confusion matrix**

| actual \ predicted | drainage | electricity | other | road | safety | waste | water |
|---|---|---|---|---|---|---|---|
| **drainage** | 162 | 0 | 0 | 0 | 0 | 0 | 0 |
| **electricity** | 0 | 172 | 0 | 0 | 0 | 0 | 0 |
| **other** | 0 | 0 | 181 | 0 | 0 | 0 | 0 |
| **road** | 0 | 0 | 0 | 176 | 0 | 0 | 0 |
| **safety** | 0 | 0 | 0 | 0 | 142 | 0 | 0 |
| **waste** | 0 | 0 | 0 | 0 | 0 | 175 | 0 |
| **water** | 0 | 0 | 0 | 0 | 0 | 0 | 192 |

**Misclassified examples**

_No misclassifications in the held-out split._

---

## 4. Priority classifier

TF-IDF + 9 engineered features (urgency-term ratio, scale terms, negation, stated
duration, length, punctuation intensity, capitalisation) → `LogisticRegression`.

- **Accuracy:** 0.962
- **Macro F1:** 0.962
- **Weighted F1:** 0.962
- **5-fold CV accuracy (train set):** 0.953

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| critical | 0.989 | 0.961 | 0.975 | 284 |
| high | 0.931 | 0.950 | 0.940 | 299 |
| low | 0.978 | 0.994 | 0.986 | 313 |
| medium | 0.950 | 0.941 | 0.946 | 304 |

**Confusion matrix**

| actual \ predicted | critical | high | low | medium |
|---|---|---|---|---|
| **critical** | 273 | 10 | 0 | 1 |
| **high** | 3 | 284 | 0 | 12 |
| **low** | 0 | 0 | 311 | 2 |
| **medium** | 0 | 11 | 7 | 286 |

**Misclassified examples**

| complaint | expected | predicted |
|---|---|---|
| Kindly look into this - a large crater has formed on the road near the park gate since yesterday. Hoping for a… | `critical` | `high` |
| Sir/Madam, the school crossing has no signage at MG Road since yesterday. | `medium` | `high` |
| We have had no water supply at the corner of the lane for several weeks. | `critical` | `medium` |
| Reporting an issue - the dustbin has been broken on the road leading to the hospital for more than a month. Ho… | `medium` | `low` |
| the open drain smells terrible near the community hall. hoping for a quick response. | `high` | `medium` |
| KINDLY LOOK INTO THIS - THE DRAIN IS COMPLETELY BLOCKED IN GANDHI NAGAR FOR OVER TEN DAYS. | `medium` | `high` |

Priority errors are overwhelmingly *adjacent* — High predicted as Critical, Medium as
High. That is the failure mode you want: the model rarely calls a critical safety hazard
"low", it just occasionally over- or under-escalates by one band, which a dispatcher
corrects in one click.

---

## 5. The number that actually matters — unseen phrasings

The random split above is **inflated, and you should not quote it.** Every sample is
generated from a shared pool of 78 issue phrasings, so the same phrasing
appears in both halves and the classifier only has to memorise it. That is textbook data
leakage, and it is why the category score above is near-perfect.

This second evaluation partitions the *phrasing pools themselves*:
20 issue phrasings and a third of the impact clauses are
withheld from training entirely, and the 1,500-sample test set is
generated exclusively from them. The model is asked to recognise ways of describing a
problem it has genuinely never read.

| metric | random split | **unseen phrasings** |
|---|---|---|
| Category accuracy | 1.000 | **0.821** |
| Category macro F1 | 1.000 | **0.824** |
| Priority accuracy | 0.962 | **0.583** |
| Priority macro F1 | 0.962 | **0.585** |
| Priority within one band | — | **0.843** |

### What the lexicon buys

The classifier does not work alone. Category prediction blends the model's probabilities
with a curated domain lexicon (`app/ml/lexicon.py`) at weight
α = 0.60, chosen by sweeping both evaluation regimes
(`python -m app.ml.tune_blend`). On unseen phrasings:

| system | category accuracy |
|---|---|
| Classifier alone | 0.525 |
| **Classifier + lexicon blend** | **0.821** |

That gap is the whole argument for the hybrid. Trained on only
58 phrasing patterns, the classifier has no representation
for a word like "nallah" or "culvert" if training never contained it. The lexicon does,
because civic vocabulary is a closed, well-understood domain — which is exactly the
situation where a curated word list beats a model starved of data, and exactly the
situation you rarely get in open-domain NLP.

**Caveat on that number.** The lexicon was authored with sight of the full phrasing pool,
so it covers some held-out vocabulary it would not have covered had it been written
blind. The blended figure is therefore optimistic too — treat the gap as real and its
exact size as an upper bound. This is also why the shipped α is 0.60 rather than the
measured optimum of 0.65.

**Read priority this way.** 84.3% of priority
predictions land within one band of correct, meaning the model rarely mistakes a
critical hazard for a low-priority nuisance; it mostly over- or under-escalates by one
step, which an administrator fixes in a single click.

**Category errors on unseen phrasings**

| complaint | expected | predicted |
|---|---|---|
| This is to inform you that the school crossing has no signage for more than a month. This has been a recurring… | `safety` | `road` |
| Sir/Madam, the school crossing has no signage on the road leading to the hospital. Nearly fifty families are a… | `safety` | `road` |
| The school crossing has no signage. Nearly fifty families are affected by this problem. | `safety` | `road` |
| Illegal parking is blocking emergency access on the road leading to the hospital since Monday. Nearly fifty fa… | `safety` | `road` |
| Reporting an issue - the meter box is open and expsed in Shivaji Nagar The situation is unsafe and spreading r… | `electricity` | `safety` |

---

## 6. Limitations — read this before trusting the numbers

1. **The training data is generated, not observed.** Even the unseen-phrasing scores in
   §5 measure generalisation *within one generator's idea* of how citizens write. Real
   municipal complaints contain phrasings, place names and problems this generator never
   produced, so real-world accuracy will be lower still. Treat §5 as an upper bound and
   the live admin-override rate as the truth.
2. **English-centric.** A small share of code-mixed samples is included, but complaints
   written mostly in Hindi, Marathi, Tamil or another Indian language will classify
   poorly. Character n-grams help with spelling, not with a different language.
3. **Priority is inferred from language, not from ground truth.** The model reads how
   alarmed the writer sounds. A calm, factual report of a genuinely dangerous problem
   will be under-prioritised; an angry report of a minor one will be over-prioritised.
4. **No fact verification.** The system triages *claims*. It cannot tell whether a
   reported leak exists, and it will confidently classify a fabricated complaint.
5. **Duplicate detection is lexical.** Cosine similarity over TF-IDF catches
   "overflowing bin at MG Road" vs "garbage bin overflowing MG Road", but misses two
   descriptions of the same problem that share no vocabulary.
6. **Class imbalance in the wild.** The training set is roughly balanced across
   priorities; real complaint streams are dominated by Medium. Calibrated confidences
   will drift accordingly, which is why the dashboard tracks the admin override rate as
   the honest real-world accuracy signal.

## 7. How to reproduce

```bash
cd backend
python -m app.ml.train --samples 6000 --seed 42
```

Deterministic given the same seed and library versions. The override rate on
`/api/v1/analytics/overview` shows how often administrators actually correct the model
once it is live — that number, not the table above, is the one to quote after launch.
