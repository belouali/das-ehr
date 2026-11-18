"""Documentation angular separation (DAS): how differently hospitals spread the evidence for a condition across
EHR sources, from aggregate counts only."""
from .vectors import LOBES, lobe_codes, lobe_counts, documentation_vectors
from .angle import das, das_matrix, mean_das_matrix
from .noise import split_half_noise, beyond_noise
from .select import rank_candidates, nearest, farthest

__all__ = ["LOBES", "lobe_codes", "lobe_counts", "documentation_vectors", "das", "das_matrix", "mean_das_matrix",
           "split_half_noise", "beyond_noise", "rank_candidates", "nearest", "farthest"]
__version__ = "1.0.0"
