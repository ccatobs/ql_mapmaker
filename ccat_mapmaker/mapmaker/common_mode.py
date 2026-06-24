# ============================================================================ #
# common_mode.py
#
# Common-mode noise subtraction (naive and iterative).
#
# At any moment all detectors see the same atmosphere but different sky pixels.
# Taking the mean across detectors estimates the atmosphere; subtracting it
# leaves each detector's individual sky signal + uncorrelated noise.
#
# The complication: bright sources bleed into the mean and get partially
# subtracted. The iterative approach fixes this by predicting the sky signal
# from a previous map and removing it before estimating the atmosphere.
# ============================================================================ #

import numpy as np


# Adapted from Jonah Lee, https://github.com/jonahjlee/blasttng-to-g3, g3_utils/signal.py, remove_common_mode
def estimate_common_mode(tod: np.ndarray) -> np.ndarray:
    """Mean across detectors at each time sample : naive atmosphere estimate."""
    return np.nanmean(tod, axis=1)


def subtract_common_mode(tod: np.ndarray,
                         common_mode: np.ndarray) -> np.ndarray:
    """Subtract the common-mode estimate from every detector's timestream."""
    return tod - common_mode[:, np.newaxis]


# Adapted from Jonah Lee, https://github.com/jonahjlee/blasttng-to-g3, g3_utils/signal.py, azelToMapPix + common_mode_iter
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


# Adapted from Jonah Lee, external/blasttng-to-g3/g3_utils/signal.py, common_mode_iter
def iterate_common_mode(tod: np.ndarray,
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
    residuals    = tod - ast_estimate
    common_mode  = np.nanmean(residuals, axis=1)
    return tod - common_mode[:, np.newaxis]
