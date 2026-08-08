"""Artifact filenames, shared by the trainer and the runtime analyzer.

Kept in its own module so ``MLAnalyzer`` does not have to import ``app.ml.train`` just
to learn a filename — that would drag scikit-learn's training APIs
(``model_selection``, ``calibration``, ``metrics``) into every cold start for no reason.
"""

from __future__ import annotations

CATEGORY_MODEL_FILE = "category_model.joblib"
PRIORITY_MODEL_FILE = "priority_model.joblib"
SIMILARITY_MODEL_FILE = "similarity_vectorizer.joblib"
METADATA_FILE = "metadata.json"
