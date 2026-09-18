#!/usr/bin/env python3
"""
colab_speedups.py -- three surgical patches that cut CALI-PRED's Colab
runtime from hours to minutes. Import and call apply_all() before running
pipeline.py, or let rerun_for_paper.py do it.

WHERE THE TIME GOES
-------------------
ImputationReliabilityEngine.compute_ensemble() is called once per window.
Every call constructs a fresh _SAITSStub and a fresh _BRITSStub, moves both
to the GPU, builds two Adam optimizers, and trains each for epochs=30 steps
on a tensor of shape (1, 60, 15).

At stride 100 the MetroPT split is 15,169 windows, so per seed that is

    15,169 windows x 2 models x 30 steps  =  910,140 forward+backward passes
    15,169 windows x 2 models             =   30,338 module builds + H2D copies
                                          +   30,338 optimizer constructions

all on tensors of 900 elements. A T4 does not help here. A (1,60,15) tensor
with d_model=64 does not come close to saturating the device, so wall-clock
is dominated by kernel-launch latency, CUDA allocation, and Python overhead.
The GPU spends most of the run idle.

THE PATCHES
-----------
1. MODEL REUSE (iri_module.py)
   Build each stub once, then call reset_parameters() per window instead of
   re-constructing and re-transferring. Each window still starts from a fresh
   random initialization, so the method is unchanged -- but 30,338 module
   builds and host-to-device copies per seed collapse to two.
   NOTE: the RNG consumption pattern differs from the original, so per-window
   initializations will not match the old run bit-for-bit. Both are random
   draws from the same distribution; nothing about the method changes.

2. CPU FOR THE IRI ENSEMBLE (iri_module.py)
   At this tensor size CPU generally beats GPU, because there is no kernel
   launch overhead to amortize. Controlled by CALIPRED_IRI_DEVICE
   (default "cpu"); set it to "cuda" to compare. This also leaves the GPU
   free for the Transformer, which is big enough to actually use it.

3. PRECOMPUTE CACHE (pipeline.py)
   DTI and the imputed sequences depend on the data and the corruption draw,
   not on the forecaster's initialization. Patch 3 fixes the corruption seed
   across runs (CALIPRED_CORRUPTION_SEED, default 12345) and caches each
   split's precompute to disk, so seeds 2 and 3 reuse what seed 1 computed.
   Fixing the corruption draw also makes the multi-seed experiment cleaner:
   the seed-to-seed spread the paper reports then measures initialization
   variance alone, rather than initialization and missingness variance mixed.
   Pass --per-seed-corruption to rerun_for_paper.py to restore the old
   behaviour (and lose the cache).

VERIFY BEFORE YOU TRUST IT
--------------------------
These patches are text substitutions checked for syntax, not for runtime
behaviour -- this machine has no PyTorch. Always smoke-test first:

    !python pipeline.py --max-windows 200 --epochs 2 --data-path ...

and confirm it completes and the logged DTI distribution looks sane.
"""
from __future__ import annotations
import os
import re
import shutil
import sys

CACHE_DIR = os.environ.get("CALIPRED_CACHE_DIR", "precompute_cache")


def _backup(path: str) -> None:
    b = path + ".prespeed"
    if not os.path.exists(b):
        shutil.copy2(path, b)


def _say(msg: str) -> None:
    print(f"[speedup] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Patch 1 + 2 : iri_module.py
# ---------------------------------------------------------------------------
_REUSE_HELPER = '''

def _calipred_reset_(module):
    """Re-randomize a module in place, so a cached instance behaves like a
    freshly constructed one without another allocation or host-to-device copy."""
    for sub in module.modules():
        if hasattr(sub, "reset_parameters"):
            sub.reset_parameters()
    return module
'''


def patch_iri(path: str = "iri_module.py") -> None:
    _backup(path)
    src = open(path, encoding="utf-8").read()

    if "_calipred_reset_" not in src:
        anchor = "class _SAITSStub("
        if anchor not in src:
            sys.exit("PATCH 1 FAILED: _SAITSStub not found.")
        src = src.replace(anchor, _REUSE_HELPER.lstrip("\n") + "\n\n" + anchor, 1)

    old = ("        saits = _SAITSStub(self.n_features, d_model=self.d_model).to(self.device)\n"
           "        brits = _BRITSStub(self.n_features, hidden_size=self.d_model).to(self.device)")
    new = ("        if getattr(self, \"_calipred_cached\", None) is None:\n"
           "            self._calipred_cached = (\n"
           "                _SAITSStub(self.n_features, d_model=self.d_model).to(self.device),\n"
           "                _BRITSStub(self.n_features, hidden_size=self.d_model).to(self.device),\n"
           "            )\n"
           "        saits, brits = self._calipred_cached\n"
           "        _calipred_reset_(saits)\n"
           "        _calipred_reset_(brits)")
    n = src.count(old)
    if n == 0 and "_calipred_cached" not in src:
        sys.exit("PATCH 1 FAILED: could not find the per-window model construction.")
    if n:
        src = src.replace(old, new, 1)
        _say(f"{path}: models are now built once and re-randomized per window")

    # Patch 2: honour CALIPRED_IRI_DEVICE, defaulting to CPU.
    old_dev = ('            device if device is not None else '
               '("cuda" if torch.cuda.is_available() else "cpu")')
    new_dev = ('            device if device is not None else '
               'os.environ.get("CALIPRED_IRI_DEVICE", "cpu")')
    if old_dev in src:
        src = src.replace(old_dev, new_dev, 1)
        if not re.search(r"^import os$", src, re.M):
            src = re.sub(r"^(import .*?)$", r"import os\n\1", src, count=1, flags=re.M)
        _say(f"{path}: IRI ensemble device now defaults to CPU "
             f"(override with CALIPRED_IRI_DEVICE)")

    open(path, "w", encoding="utf-8").write(src)
    import ast
    ast.parse(open(path, encoding="utf-8").read())


# ---------------------------------------------------------------------------
# Patch 3 : pipeline.py precompute cache + fixed corruption seed
# ---------------------------------------------------------------------------
_CACHE_WRAPPER = '''

def _calipred_cseed():
    """The seed governing everything the precompute depends on: the split,
    the imputation ensemble's initialization, the corruption draw and the
    severity sampler. Pinned so that DTI and the imputed sequences are a
    function of the data alone, not of the forecaster's initialization."""
    return int(os.environ.get("CALIPRED_CORRUPTION_SEED", "12345"))


def _calipred_cache_key(dataset, missing_rate, batch_size):
    """Identify a split by its length and its first/last window contents."""
    import hashlib
    import numpy as _np
    h = hashlib.sha1()
    h.update(str(len(dataset)).encode())
    h.update(str(missing_rate).encode())
    h.update(str(batch_size).encode())
    h.update(os.environ.get("CALIPRED_CORRUPTION_SEED", "12345").encode())
    for idx in (0, len(dataset) - 1):
        try:
            item = dataset[idx]
            arr = item[0] if isinstance(item, (tuple, list)) else item
            h.update(_np.ascontiguousarray(
                arr.numpy() if hasattr(arr, "numpy") else arr).tobytes()[:65536])
        except Exception:
            pass
    return h.hexdigest()[:16]


def precompute_trust_and_imputed(*args, **kwargs):
    """Disk-cached wrapper. DTI and the imputed sequences depend on the data
    and the corruption draw, not on the forecaster seed, so the second and
    third seeds of a multi-seed run can reuse the first seed's work."""
    import numpy as _np
    cache_dir = os.environ.get("CALIPRED_CACHE_DIR", "precompute_cache")
    if os.environ.get("CALIPRED_DISABLE_CACHE"):
        return _precompute_trust_and_imputed_uncached(*args, **kwargs)
    os.makedirs(cache_dir, exist_ok=True)
    dataset = args[0] if args else kwargs.get("dataset")
    key = _calipred_cache_key(dataset,
                             kwargs.get("missing_rate", 0.15),
                             kwargs.get("batch_size", 128))
    path = os.path.join(cache_dir, f"precompute_{key}.npz")
    if os.path.exists(path):
        z = _np.load(path)
        logger.info("Precompute cache HIT  (%s, %d windows) -- skipping %d model fits.",
                    path, len(dataset), len(dataset) * 2)
        return z["dti"], z["imputed"]
    logger.info("Precompute cache MISS (%s) -- computing.", path)
    dti, imputed = _precompute_trust_and_imputed_uncached(*args, **kwargs)
    _np.savez_compressed(path, dti=dti, imputed=imputed)
    logger.info("Precompute cached -> %s", path)
    return dti, imputed
'''


def patch_pipeline(path: str = "pipeline.py") -> None:
    _backup(path)
    src = open(path, encoding="utf-8").read()

    if "_precompute_trust_and_imputed_uncached" not in src:
        if "def precompute_trust_and_imputed(" not in src:
            sys.exit("PATCH 3 FAILED: precompute_trust_and_imputed not found.")
        src = src.replace("def precompute_trust_and_imputed(",
                          "def _precompute_trust_and_imputed_uncached(", 1)
        # Insert the wrapper after the original function, i.e. just before the
        # next top-level definition.
        m = re.search(r"\n(?=(?:def |class )\w)", src[src.index(
            "def _precompute_trust_and_imputed_uncached("):])
        if not m:
            sys.exit("PATCH 3 FAILED: no insertion point after the original function.")
        pos = src.index("def _precompute_trust_and_imputed_uncached(") + m.start() + 1
        src = src[:pos] + _CACHE_WRAPPER + src[pos:]
        _say(f"{path}: precompute is now disk-cached across seeds")

    # The missingness SEVERITY samplers are seeded from args.seed too, so they
    # must also be pinned or a cached precompute would not match what an
    # uncached run of the same seed would produce.
    old_s = ("    train_val_sampler = make_severity_sampler(\n"
             "        clean_fraction=args.clean_fraction,\n"
             "        max_severity=args.max_severity,\n"
             "        random_state=args.seed,\n"
             "    )\n"
             "    test_sampler = make_severity_sampler(\n"
             "        clean_fraction=args.clean_fraction,\n"
             "        max_severity=args.max_severity,\n"
             "        random_state=args.seed + 58,\n"
             "    )")
    new_s = ("    train_val_sampler = make_severity_sampler(\n"
             "        clean_fraction=args.clean_fraction,\n"
             "        max_severity=args.max_severity,\n"
             "        random_state=_calipred_cseed(),\n"
             "    )\n"
             "    test_sampler = make_severity_sampler(\n"
             "        clean_fraction=args.clean_fraction,\n"
             "        max_severity=args.max_severity,\n"
             "        random_state=_calipred_cseed() + 58,\n"
             "    )")
    if old_s in src:
        src = src.replace(old_s, new_s, 1)
        _say(f"{path}: missingness severity samplers pinned")
    elif "random_state=_calipred_cseed()," not in src:
        _say(f"WARNING: {path}: severity samplers not found; run with "
             f"CALIPRED_DISABLE_CACHE=1 until this is checked by hand.")

    # The imputation ensemble's own seed drives its initialization and holdout
    # draws, so it feeds straight into DTI. It must be pinned or a cache entry
    # written under seed 42 would be wrong for seeds 123 and 456.
    old_i = ("    iri_engine = ImputationReliabilityEngine(\n"
             "        n_features=n_features,\n"
             "        epochs=30,  # lighter for pipeline use\n"
             "        holdout_frac=0.15,\n"
             "        random_state=args.seed,\n"
             "    )")
    new_i = ("    iri_engine = ImputationReliabilityEngine(\n"
             "        n_features=n_features,\n"
             "        epochs=int(os.environ.get(\"CALIPRED_IRI_EPOCHS\", \"30\")),\n"
             "        holdout_frac=0.15,\n"
             "        random_state=_calipred_cseed(),\n"
             "    )")
    if old_i in src:
        src = src.replace(old_i, new_i, 1)
        _say(f"{path}: imputation ensemble pinned; epochs via CALIPRED_IRI_EPOCHS")
    elif "CALIPRED_IRI_EPOCHS" not in src:
        _say(f"WARNING: {path}: IRI engine construction not found -- the cache "
             f"may be unsafe. Run with CALIPRED_DISABLE_CACHE=1.")

    # Fixed corruption seed so the cache is reachable from every model seed.
    old_c = "corruption_loader = IndustrialDataLoader(random_state=args.seed)"
    new_c = ('corruption_loader = IndustrialDataLoader(\n'
             '        random_state=int(os.environ.get("CALIPRED_CORRUPTION_SEED", "12345")))')
    if old_c in src:
        src = src.replace(old_c, new_c, 1)
        _say(f"{path}: corruption draw fixed across seeds "
             f"(CALIPRED_CORRUPTION_SEED, default 12345)")

    if not re.search(r"^import os$", src, re.M):
        src = re.sub(r"^(import .*?)$", r"import os\n\1", src, count=1, flags=re.M)

    open(path, "w", encoding="utf-8").write(src)
    import ast
    ast.parse(open(path, encoding="utf-8").read())


# ---------------------------------------------------------------------------
# Patch 4 : refuse to silently substitute synthetic data
# ---------------------------------------------------------------------------
def patch_no_mock(path: str = "data_loader.py") -> None:
    """IndustrialDataLoader catches a missing CSV, logs an error, and then
    carries on with 2,000 timesteps of synthetic mock data. The run completes
    and produces numbers that look entirely ordinary. Nothing downstream --
    not the metrics, not the figures, not the paper -- can tell the difference.

    After this patch a missing or truncated dataset raises instead. Set
    CALIPRED_ALLOW_MOCK=1 if you genuinely want a synthetic dry run."""
    _backup(path)
    src = open(path, encoding="utf-8", newline="").read()

    if "CALIPRED_ALLOW_MOCK" in src:
        _say(f"{path}: already guarded")
        return

    guard = (
        '            if os.environ.get("CALIPRED_ALLOW_MOCK") != "1":\\1'
        '                raise FileNotFoundError(\\1'
        '                    f"Refusing to fall back to synthetic mock data for "\\1'
        '                    f"{file_path!r}. A run on mock data produces numbers "\\1'
        '                    f"indistinguishable from real results. Fetch the dataset "\\1'
        '                    f"(python download_data.py --dataset metropt), or set "\\1'
        '                    f"CALIPRED_ALLOW_MOCK=1 for a deliberate dry run."\\1'
        '                ) from exc\\1'
    )
    pat = re.compile(r"(            is_mock = True)(\r?\n)")
    if not pat.search(src):
        sys.exit("PATCH 4 FAILED: could not find the mock fallback branch.")
    src = pat.sub(lambda m: m.group(1) + m.group(2)
                  + guard.replace("\\1", m.group(2)), src, count=1)

    # A truncated CSV only warns; make that fatal too.
    pat2 = re.compile(
        r"(            if len\(df\) < 5000:\r?\n)"
        r"(                logger\.warning\([^\n]*\r?\n)")
    m2 = pat2.search(src)
    if m2:
        nl = "\r\n" if "\r\n" in m2.group(0) else "\n"
        extra = (f'                if os.environ.get("CALIPRED_ALLOW_MOCK") != "1":{nl}'
                 f'                    raise ValueError({nl}'
                 f'                        f"{{file_path!r}} has only {{len(df)}} rows; "{nl}'
                 f'                        f"the real MetroPT record has ~1.5M. Refusing to run."{nl}'
                 f'                    ){nl}')
        src = src[:m2.end()] + extra + src[m2.end():]

    if not re.search(r"^import os\s*$", src, re.M):
        src = re.sub(r"^(import .*?)$", r"import os\n\1", src, count=1, flags=re.M)

    open(path, "w", encoding="utf-8", newline="").write(src)
    import ast
    ast.parse(open(path, encoding="utf-8").read())
    _say(f"{path}: mock-data fallback now raises "
         f"(override with CALIPRED_ALLOW_MOCK=1)")


def apply_all(iri="iri_module.py", pipeline="pipeline.py",
              loader="data_loader.py") -> None:
    patch_iri(iri)
    patch_pipeline(pipeline)
    patch_no_mock(loader)
    _say("all patches applied; .prespeed backups kept alongside each file")


if __name__ == "__main__":
    apply_all()
