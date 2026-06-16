"""Learned (neural) Builder for partial binary relations.

This subpackage uses torch. It is intentionally NOT imported by
``logical_gans.modelbuilder.__init__`` so that importing the core kernel
stays torch-free. Import these modules explicitly (the CLI does so lazily).

Milestone scope: a neural policy proposes edits to ONE binary relation; the
symbolic Devil verifies every edit. No functions/constants, MCTS, or LLM.
"""
