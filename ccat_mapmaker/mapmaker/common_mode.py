# ============================================================================ #
# common_mode.py
#
# James Burgoyne, jburgoyne@phas.ubc.ca
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT August 2026
#
# Common-mode noise subtraction (naive and iterative).
# ============================================================================ #

import numpy as np


def _mean_over_valid(tod: np.ndarray, flag_mask: np.ndarray) -> np.ndarray:
    """Fast axis-1 mean that ignores invalid samples/NaNs without allocating more arrays than necessary."""
    valid = np.isfinite(flag_mask)
    if valid.all():
        return tod.mean(axis=1, dtype=np.float64)
    if not valid.any():
        return np.zeros(tod.shape[0], dtype=float)

    masked = np.where(valid, tod, 0.0)
    counts = valid.sum(axis=1)
    summed = masked.sum(axis=1)
    return np.divide(summed, counts, out=np.zeros_like(summed, dtype=float), where=counts > 0)


# ============================================================================ #
# estimate_common_mode
# ============================================================================ #
def estimate_common_mode(tod: np.ndarray, flag_mask: np.ndarray) -> np.ndarray:
    """Mean across detectors at each time sample : naive atmosphere estimate."""
    # TODO: The detectors are pointing at different parts of the sky
    # so this only removes signal common at one time.
    # Elevation component of atmosphere not perfectly removed with this.
    # Can we remove it some other way?
    return _mean_over_valid(tod, flag_mask)


# ============================================================================ #
# subtract_common_mode
# ============================================================================ #
def subtract_common_mode(tod: np.ndarray,
                         common_mode: np.ndarray) -> np.ndarray:
    """Subtract the common-mode estimate from every detector's timestream."""
    return tod - common_mode[:, np.newaxis]


# ============================================================================ #
# lookup_map_signal
# ============================================================================ #
def lookup_map_signal(combined_map: np.ndarray,
                      ra: np.ndarray, dec: np.ndarray,
                      ra_edges: np.ndarray, dec_edges: np.ndarray) -> np.ndarray:
    """
    For each detector at each time sample, look up the sky signal from the
    current map at that detector's pointing position.

    Unobserved pixels (NaN) return 0, no prediction where we have no data.
    Vectorised over all detectors and time samples via searchsorted.
    """
    ny, nx = combined_map.shape

    ix = np.searchsorted(ra_edges,  ra,  side='right') - 1
    iy = np.searchsorted(dec_edges, dec, side='right') - 1

    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    vals = combined_map[iy, ix]
    return np.where(np.isfinite(vals), vals, 0.0)


# ============================================================================ #
# iterate_common_mode
# ============================================================================ #
def iterate_common_mode(tod: np.ndarray, flag_mask: np.ndarray,
                        ra: np.ndarray, dec: np.ndarray,
                        combined_map: np.ndarray,
                        ra_edges: np.ndarray,
                        dec_edges: np.ndarray) -> np.ndarray:
    """
    One iteration of sky-informed common-mode removal.

    1. Predict sky signal per detector from the previous map.
    2. Subtract prediction from tod -> residuals (mostly atmosphere).
    3. Mean of residuals -> cleaner atmosphere estimate than naive mean.
    4. Subtract atmosphere estimate from original tod (preserves sky signal).
    """
    ast_estimate = lookup_map_signal(combined_map, ra, dec, ra_edges, dec_edges)
    residuals = (tod - ast_estimate)
    common_mode = _mean_over_valid(residuals, flag_mask)
    return tod - common_mode[:, np.newaxis]
