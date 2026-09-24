# ============================================================================ #
# binning.py
#
# James Burgoyne, jburgoyne@phas.ubc.ca
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT August 2026
#
# Sky map pixelization: bin detector timestreams into a 2-D map.
#
# For each time sample, we know where the detector was pointing and what signal
# it recorded. We divide the sky into a rectangular pixel grid and accumulate
# signal from all samples that fell in each pixel. The map value per pixel is
# the average of all signal samples that landed there, computed via numpy's histogram2d.
# ============================================================================ #

import numpy as np


# ============================================================================ #
# make_map_edges
# ============================================================================ #
def make_map_edges(ra0_deg: float, dec0_deg: float,
                   xlen_deg: float, ylen_deg: float,
                   res_deg: float):
    """
    Create RA and Dec bin-edge arrays defining the pixel grid.

    Grid is centred at (ra0_deg, dec0_deg), spans xlen_deg x ylen_deg degrees,
    with square pixels of size res_deg. Returns (nx+1,) and (ny+1,) edge arrays
    where nx = int(xlen_deg / res_deg), ny = int(ylen_deg / res_deg).
    """
    nx = int(xlen_deg / res_deg)
    ny = int(ylen_deg / res_deg)

    ra_edges  = np.linspace(ra0_deg  - xlen_deg / 2,
                            ra0_deg  + xlen_deg / 2, nx + 1)
    dec_edges = np.linspace(dec0_deg - ylen_deg / 2,
                            dec0_deg + ylen_deg / 2, ny + 1)
    return ra_edges, dec_edges


# ============================================================================ #
# bin_detector
# This is only really used for the per detector maps right now
# ============================================================================ #
def bin_detector(tod_1d: np.ndarray, flag_mask_1d: np.ndarray,
                 ra_1d: np.ndarray, dec_1d: np.ndarray,
                 ra_edges: np.ndarray, dec_edges: np.ndarray,
                 dtype=np.float32):
    """
    Bin one detector's timestream into a 2-D pixel map.

    Runs histogram2d twice: once with signal as weights to get summed signal
    per pixel (data), once without to get sample count per pixel (hits).
    Dividing data by hits gives the average signal per pixel.
    """
    flags = np.copy(flag_mask_1d)
    flags[np.isnan(flags)] = 0
    data, _, _ = np.histogram2d(dec_1d, ra_1d,
                                bins=[dec_edges, ra_edges],
                                weights=tod_1d*flags)
    hits, _, _ = np.histogram2d(dec_1d[np.isnan(flag_mask_1d) == False], ra_1d[np.isnan(flag_mask_1d) == False],
                                bins=[dec_edges, ra_edges])
    return data.astype(dtype), hits.astype(dtype)


# ============================================================================ #
# bin_chunk
# ============================================================================ #
def bin_chunk(signal: np.ndarray, flag_mask: np.ndarray,
              ra: np.ndarray, dec: np.ndarray,
              ra_edges: np.ndarray, dec_edges: np.ndarray,
              weights: np.ndarray = None):
    """
    Bin one Chunk's signal into a 2-D pixel map.

    Uses a direct integer binning path for regular map grids instead of
    repeatedly calling histogram2d, which dominates the runtime in the
    streaming pass when many chunks are processed.

    signal  : (n_samps, n_dets) cleaned signal
    flags_mask: (n_sampls, n_dets) mask of signal
    ra, dec : (n_samps, n_dets) per-detector pointing in degrees
    weights : (n_dets,) optional per-detector weights; defaults to uniform
    Returns data, hits, and sumsq (weighted sum of signal**2), all shape (ny, nx).
    sumsq accumulates per-pixel variance (E[x^2] - E[x]^2) across chunks for a real RMS noise map.

    Weighting is per-sample, not per-detector: a flagged sample drops only
    itself from the map, not its detector's whole chunk (matches how
    bin_detector already excludes samples one at a time).
    """
    n_dets = signal.shape[1]
    det_w = np.ones(n_dets, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    flags = np.nan_to_num(flag_mask, nan=0.0, posinf=0.0, neginf=0.0)
    w = flags * det_w[np.newaxis, :]  # (n_samps, n_dets): 0 where flagged, det_w elsewhere

    ra_flat = np.ascontiguousarray(ra).ravel()
    dec_flat = np.ascontiguousarray(dec).ravel()
    w_flat = (w).ravel()
    sig_flat = (signal * w).ravel()
    sig_sq_flat = (signal * signal * w).ravel()

    ny = dec_edges.size - 1
    nx = ra_edges.size - 1
    ra0 = float(ra_edges[0])
    dec0 = float(dec_edges[0])

    ra_step = np.diff(ra_edges)
    dec_step = np.diff(dec_edges)
    if ra_step.size and dec_step.size and np.allclose(ra_step, ra_step[0]) and np.allclose(dec_step, dec_step[0]):
        ra_bin = float(ra_step[0])
        dec_bin = float(dec_step[0])
        ix = np.floor((ra_flat - ra0) / ra_bin).astype(np.int64, copy=False)
        iy = np.floor((dec_flat - dec0) / dec_bin).astype(np.int64, copy=False)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        flat_idx = iy * nx + ix
        valid = np.isfinite(ra_flat) & np.isfinite(dec_flat) & (w_flat != 0.0)
        flat_idx = flat_idx[valid]
        if flat_idx.size == 0:
            return np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)
        data = np.bincount(flat_idx, weights=sig_flat[valid], minlength=nx * ny).reshape(ny, nx)
        hits = np.bincount(flat_idx, weights=w_flat[valid], minlength=nx * ny).reshape(ny, nx)
        sumsq = np.bincount(flat_idx, weights=sig_sq_flat[valid], minlength=nx * ny).reshape(ny, nx)
        return data, hits, sumsq

    data, _, _ = np.histogram2d(dec_flat, ra_flat,
                               bins=[dec_edges, ra_edges],
                               weights=sig_flat)
    hits, _, _ = np.histogram2d(dec_flat, ra_flat,
                               bins=[dec_edges, ra_edges],
                               weights=w_flat)
    sumsq, _, _ = np.histogram2d(dec_flat, ra_flat,
                                 bins=[dec_edges, ra_edges],
                                 weights=sig_sq_flat)
    return data, hits, sumsq