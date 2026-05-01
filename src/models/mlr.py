"""Multinomial logistic-regression model family used in the final report.

Implementation lives in ``src.pipeline.fit_predict_mlr`` so the submitted
entry point remains a single reproducible command. This module documents the
MLR variants trained by the pipeline and keeps model-family organization
visible in the submitted ``final/`` directory.
"""

VARIANTS = [
    "MLR-original-baseline-5",
    "MLR-original-18",
    "MLR-original-form",
    "MLR-original-matchup",
    "MLR-strength",
    "MLR-full",
    "MLR-recent-no-elo",
]

