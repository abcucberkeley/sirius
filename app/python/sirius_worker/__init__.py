"""SIRIUS compute worker.

A small TCP service the SIRIUS desktop application talks to for work that
lives in Python -- Torch segmentation models above all -- and the same
service that runs on a cluster node for the HPC backend. Protocol: see
``protocol.py`` (mirrors ``app/core/rpc.hpp``); step implementations: see
``sirius.workbench`` (located by ``steps.py``).

    python -m sirius_worker --host 127.0.0.1 --port 0 --token X --device auto
"""

# Same literal as `project(... VERSION ...)` in the top-level CMakeLists.txt, which
# is canonical, and as pyproject.toml; tools/check_versions.py keeps them in step.
__version__ = "0.1.0"

# What the worker imports, by import name, with the distribution that provides
# it. REQUIRED is what it cannot start without (``python -m sirius_worker``
# exits 3 with a ``missing_packages`` line when one is absent) and is what
# ../requirements.txt lists; SIRIUS installs it into its own Python
# environment. OPTIONAL is what single steps and the model hub use; SIRIUS
# installs these only when asked (scipy and scikit-image are
# ../requirements-extra.txt). app/core/python_env.cpp mirrors OPTIONAL's
# values (pyenv::optionalDistributions), and a C++ test checks it does.
REQUIRED = {"numpy": "numpy"}
OPTIONAL = {
    "scipy": "scipy",
    "skimage": "scikit-image",
    "torch": "torch",
    "huggingface_hub": "huggingface_hub",
    "onnxruntime": "onnxruntime",
    "cellpose": "cellpose",
    "micro_sam": "micro_sam",
    "btrack": "btrack",
    "tifffile": "tifffile",
}

__all__ = ["OPTIONAL", "REQUIRED", "__version__"]
