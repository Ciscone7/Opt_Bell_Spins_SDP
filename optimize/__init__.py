# ---------------------------------------------------------------------------
# Core exports: these work with minimal dependencies (numpy, cvxpy, tqdm)
# ---------------------------------------------------------------------------
from .montecarlo import simulated_annealing, parallel_tempering
#from ..src.optimalsdp.bell_utils import MomentMatrix_Bell, Behaviour

# results_manager requires h5py, but it's commonly installed
# Wrap in try/except so the package is still importable without it
# try:
#     from ..src.optimalsdp.results_manager import (
#         check_simulation_status,
#         load_results_from_config,
#         load_specific_params,
#         results_to_dataframe,
#     )
# except ImportError:
#     # h5py not installed - results_manager unavailable at package level
#     pass

# ---------------------------------------------------------------------------
# Optional modules are NOT imported here to avoid pulling in heavy deps.
# Users should import them explicitly:
#   from optimalsdp.bayesian import bayesian      # needs scikit-learn
#   from optimalsdp.rbm import RBM, RBMTrainer    # needs jax, equinox, optax
# ---------------------------------------------------------------------------

__all__ = [
    # montecarlo
    "simulated_annealing",
    "parallel_tempering",
    # bell_utils
    #"MomentMatrix_Bell",
    #"Behaviour",
    # results_manager (may not be available)
    #"check_simulation_status",
    #"load_results_from_config",
    #"load_specific_params",
    #"results_to_dataframe",
]
