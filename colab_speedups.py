"""Deprecated speedup hooks.

The historical implementation monkey-patched source files at runtime and
changed the imputation semantics. It is intentionally disabled so experiments
run the checked-in implementation without hidden source mutations.
"""

def apply_all():
    print(
        "[CALI-PRED] Runtime source patching is disabled for reproducibility. "
        "Run pipeline.py directly; use explicit CLI options for smoke tests."
    )

if __name__ == "__main__":
    apply_all()
