"""Measurement and operational tooling that orchestrates Carmel services.

Modules here are runnable harnesses, not part of the closed-loop pipeline: they
drive the production services over real inputs to measure behaviour. They ship
with tests and are type-checked exactly as the rest of the package is, but they
never relax a pipeline guard to change what they measure.
"""
