"""
Minimal compatibility package for ML interfaces.

This package provides thin wrappers and stubs so imports like
`from ml.inference import InferenceEngine` succeed even when the
full training/serving stack isn't present. Keep implementations
minimal and import-safe.
"""
