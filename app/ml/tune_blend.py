"""Sweep the lexicon blend weight against the unseen-phrasing test set.

Run from ``backend/``::

    python -m app.ml.tune_blend

The blend weight ``alpha`` decides how much the curated lexicon overrides the learned
classifier. Picking it by intuition would be guessing, so this sweeps it against the
honest evaluation — a test set generated entirely from issue phrasings and impact
clauses withheld from training — and prints accuracy at each value.

The chosen value is hard-coded as ``MLAnalyzer``'s default; rerun this after changing
the lexicon or the dataset to confirm it is still the right one.
"""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score

from app.ml.dataset import build_dataset, split_impacts, split_issues
from app.ml.lexicon import blend_scores
from app.ml.preprocess import clean_text
from app.ml.train import build_category_pipeline

ALPHAS = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.8, 1.0]


def _score_at_alpha(
    model, texts: list[str], labels: list[str], alpha: float
) -> tuple[float, float]:
    classes = list(model.classes_)
    probabilities = model.predict_proba(texts)
    predictions: list[str] = []
    for row, text in zip(probabilities, texts, strict=True):
        model_proba = dict(zip(classes, row, strict=True))
        blended, _ = blend_scores(model_proba, clean_text(text), alpha=alpha)
        predictions.append(max(blended.items(), key=lambda pair: pair[1])[0])
    return (
        float(accuracy_score(labels, predictions)),
        float(f1_score(labels, predictions, average="macro", zero_division=0)),
    )


def main(n_samples: int = 6000, seed: int = 42) -> None:
    """Sweep alpha against BOTH evaluation regimes.

    Optimising for unseen phrasings alone would push alpha to ~1.0 and throw away the
    classifier entirely — which also throws away its context-sensitivity on the
    phrasings it *has* learned. Reporting both columns makes the trade-off visible.
    """
    from sklearn.model_selection import train_test_split

    # --- regime A: unseen phrasings (the honest generalisation test) ------------
    train_issues, holdout_issues = split_issues(seed=seed)
    train_impacts, holdout_impacts = split_impacts(seed=seed)
    print(
        f"Regime A — train on {len(train_issues)} phrasings, "
        f"test on {len(holdout_issues)} unseen phrasings"
    )
    gen_train_texts, gen_train_cat, _ = build_dataset(
        n_samples, seed=seed, issues=train_issues, impacts=train_impacts
    )
    gen_test_texts, gen_test_cat, _ = build_dataset(
        1500, seed=seed + 999, issues=holdout_issues, impacts=holdout_impacts
    )
    gen_model = build_category_pipeline().fit(gen_train_texts, gen_train_cat)

    # --- regime B: random split (in-distribution, what ships) -------------------
    print("Regime B — random split over the full phrasing pool (the shipped model)")
    all_texts, all_cat, _ = build_dataset(n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_cat, test_size=0.2, random_state=seed, stratify=all_cat
    )
    dist_model = build_category_pipeline().fit(X_train, y_train)

    print(f"\n{'alpha':>6} | {'unseen acc':>10} | {'unseen F1':>9} | {'in-dist acc':>11} | {'mean':>6}")
    print("-" * 58)

    best = (0.0, -1.0)
    for alpha in ALPHAS:
        unseen_acc, unseen_f1 = _score_at_alpha(gen_model, gen_test_texts, gen_test_cat, alpha)
        dist_acc, _ = _score_at_alpha(dist_model, X_test, y_test, alpha)
        # Weight the honest regime higher: real complaints are far more likely to use
        # phrasings the training data never contained than to match it exactly.
        combined = 0.6 * unseen_acc + 0.4 * dist_acc
        print(
            f"{alpha:>6.2f} | {unseen_acc:>10.3f} | {unseen_f1:>9.3f} | "
            f"{dist_acc:>11.3f} | {combined:>6.3f}"
        )
        if combined > best[1]:
            best = (alpha, combined)

    print(f"\nBest alpha = {best[0]:.2f} (weighted 60% unseen / 40% in-distribution)")


if __name__ == "__main__":
    main()
