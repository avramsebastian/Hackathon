"""
ml — Machine-learning pipeline
===============================

Sub-packages
------------
comunication
    :func:`~comunication.Inference.fa_inferenta_din_json` inference entry
    point and an optional FastAPI server.
entities
    Lightweight data classes (:class:`Car`, :class:`Sign`, :class:`Directions`,
    :class:`Intersections`) used for feature extraction.
learn
    Data generation, training and evaluation scripts.
generated
    Artefacts produced by the pipeline (CSV datasets, ``.pkl`` model,
    test reports).
"""
