# ============================================================================ #
# pointing.py
#
# James Burgoyne, jburgoyne@phas.ubc.ca
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT August 2026
#
# Converts telescope pointing to sky coordinates (RA, Dec) for each detector.
#
# Simulation format: boresight is a quaternion timestream (one per sample).
#   Each detector has a fixed offset quaternion from the focal plane file.
#   Combining the two gives per-detector RA/Dec.
#
# Imported by reader.py to load data
# ============================================================================ #

import numpy as np
from scipy.spatial.transform import Rotation

_Z_AXIS = np.array([0.0, 0.0, 1.0])


# ============================================================================ #
# _pointing_to_radec
# ============================================================================ #
def _pointing_to_radec(pointing: np.ndarray) -> tuple:
    """Convert (n_samps, 3) unit vectors to (ra, dec) in degrees."""
    ra  = np.degrees(np.arctan2(pointing[:, 1], pointing[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(pointing[:, 2], -1.0, 1.0)))
    return ra, dec


# ============================================================================ #
# precompute_det_directions
# ============================================================================ #
def precompute_det_directions(det_quats: np.ndarray) -> np.ndarray:
    """
    Rotate the z-axis by each detector's offset quaternion -> (n_dets, 3).

    Called once at focalplane load. Avoids recomputing per frame using:
        (R_bore * R_det).apply(z) == R_bore.apply(R_det.apply(z))

    det_quats : (n_dets, 4), vector-first (x, y, z, w).
    """
    return Rotation.from_quat(det_quats).apply(_Z_AXIS)


# ============================================================================ #
# boresight_to_radec
# ============================================================================ #
def boresight_to_radec(boresight_q: np.ndarray) -> tuple:
    """
    Boresight quaternion timestream -> (ra, dec, boresight_r).

    boresight_r is returned so callers can reuse it per detector
    without rebuilding it.

    boresight_q : (n_samps, 4), scalar-first (w, x, y, z).
    """
    # Reorder w,x,y,z -> x,y,z,w to match scipy's convention
    boresight_r = Rotation.from_quat(boresight_q[:, [1, 2, 3, 0]])
    ra, dec = _pointing_to_radec(boresight_r.apply(_Z_AXIS))
    return ra, dec, boresight_r


# ============================================================================ #
# det_radec_from_boresight
# ============================================================================ #
def det_radec_from_boresight(boresight_r, det_dir: np.ndarray) -> tuple:
    """
    Per-sample RA/Dec for one detector. Hot path — called once per detector
    per frame, reusing boresight_r from boresight_to_radec.

    boresight_r : Rotation of shape (n_samps,).
    det_dir     : (3,) precomputed direction from precompute_det_directions.
    """
    return _pointing_to_radec(boresight_r.apply(det_dir))
