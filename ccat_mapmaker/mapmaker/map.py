# ============================================================================ #
# map.py
#
# James Burgoyne, jamesrburgoyne@icloud.com
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT 2026
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
# makeMapEdges
def makeMapEdges(
        ra0_deg: float, dec0_deg: float,
        xlen_deg: float, ylen_deg: float, 
        res_deg: float):
    """Create RA and Dec bin-edge arrays defining the pixel grid.

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

    return {
        'ra_edges': ra_edges,
        'dec_edges': dec_edges,
        'nx': nx,
        'ny': ny
    }


# ============================================================================ #
# bin_detector
# This is only really used for the per detector maps right now
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
# binChunk
def binChunk(
        signal: np.ndarray, flag_mask: np.ndarray,
        ra: np.ndarray, dec: np.ndarray,
        ra_edges: np.ndarray, dec_edges: np.ndarray,
        weights: np.ndarray = None):
    """Bin one Chunk's signal into a 2-D pixel map.

    signal  : (n_samps, n_dets) cleaned signal
    flags_mask: (n_sampls, n_dets) mask of signal
    ra, dec : (n_samps, n_dets) per-detector pointing in degrees
    weights : (n_dets,) optional per-detector weights; defaults to uniform'

    Returns data, hits, and sumsq (weighted sum of signal**2), all shape (ny, nx).
    sumsq accumulates per-pixel variance (E[x^2] - E[x]^2) across chunks for a real RMS noise map.
    """

    # Initialize detector weights and sanitize flags against NaN/inf
    n_dets = signal.shape[1]
    det_w = np.ones(n_dets, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    
    # Flatten arrays early to minimize 2D temporary allocations
    sig_orig = np.ascontiguousarray(signal).ravel()
    ra_flat = np.ascontiguousarray(ra).ravel()
    dec_flat = np.ascontiguousarray(dec).ravel()
    
    # Compute weights and apply to signal streams
    flags = np.nan_to_num(flag_mask, nan=0.0, posinf=0.0, neginf=0.0)
    w_flat = (flags * det_w[np.newaxis, :]).ravel()
    
    sig_flat = sig_orig * w_flat
    sig_sq_flat = sig_orig * sig_flat  # Reuses sig_flat to avoid redundant multiplication

    # Extract grid dimensions and origin coordinates
    ny = dec_edges.size - 1
    nx = ra_edges.size - 1
    ra0 = float(ra_edges[0])
    dec0 = float(dec_edges[0])

    ra_step = np.diff(ra_edges)
    dec_step = np.diff(dec_edges)

    # Determine if the grid is regular (uniform step size)
    is_regular = (
        ra_step.size and dec_step.size and 
        np.allclose(ra_step, ra_step[0]) and 
        np.allclose(dec_step, dec_step[0])
    )

    # Compute pixel indices based on grid regularity
    if is_regular:
        ra_bin = float(ra_step[0])
        dec_bin = float(dec_step[0])
        ix = np.floor((ra_flat - ra0) / ra_bin).astype(np.int64, copy=False)
        iy = np.floor((dec_flat - dec0) / dec_bin).astype(np.int64, copy=False)
    else:
        # Fallback for irregular grids using digitize (0-indexed bin assignment)
        ix = np.digitize(ra_flat, ra_edges) - 1
        iy = np.digitize(dec_flat, dec_edges) - 1

    # Build combined validity mask: drops out-of-bounds samples instead of clipping them
    valid = (
        np.isfinite(ra_flat) & np.isfinite(dec_flat) & 
        (w_flat != 0.0) & 
        (ix >= 0) & (ix < nx) & 
        (iy >= 0) & (iy < ny)
    )

    # Early exit if no samples fall within valid grid boundaries
    if not np.any(valid):
        empty_map = np.zeros((ny, nx), dtype=float)
        return empty_map, empty_map.copy(), empty_map.copy()

    # Map 2D indices to flat 1D index array for valid samples only
    flat_idx = iy[valid] * nx + ix[valid]

    # Single-pass accumulation via bincount for all three metrics
    data = np.bincount(
        flat_idx, weights=sig_flat[valid], minlength=nx * ny
        ).reshape(ny, nx)
    hits = np.bincount(
        flat_idx, weights=w_flat[valid], minlength=nx * ny
        ).reshape(ny, nx)
    sumsq = np.bincount(
        flat_idx, weights=sig_sq_flat[valid], minlength=nx * ny
        ).reshape(ny, nx)

    return data, hits, sumsq
    return data, hits, sumsq

def binChunkPair(
        naive_signal: np.ndarray, signal: np.ndarray, flag_mask: np.ndarray,
        ra: np.ndarray, dec: np.ndarray,
        ra_edges: np.ndarray, dec_edges: np.ndarray,
        weights: np.ndarray,
        ):
    """Bin naive and weighted signals while computing pixel indices once."""
    n_dets = signal.shape[1]
    det_w = np.asarray(weights, dtype=float)
    if naive_signal.shape != signal.shape or det_w.shape != (n_dets,):
        raise ValueError("Naive/weighted signals and detector weights must have matching shapes")
    naive_flat = np.ascontiguousarray(naive_signal).ravel()
    sig_flat = np.ascontiguousarray(signal).ravel()
    ra_flat = np.ascontiguousarray(ra).ravel()
    dec_flat = np.ascontiguousarray(dec).ravel()
    flags = np.nan_to_num(flag_mask, nan=0.0, posinf=0.0, neginf=0.0).ravel()

    ny = dec_edges.size - 1
    nx = ra_edges.size - 1
    ra_step = np.diff(ra_edges)
    dec_step = np.diff(dec_edges)
    is_regular = (
        ra_step.size and dec_step.size
        and np.allclose(ra_step, ra_step[0])
        and np.allclose(dec_step, dec_step[0])
    )
    finite_pointing = np.isfinite(ra_flat) & np.isfinite(dec_flat)
    safe_ra = np.where(finite_pointing, ra_flat, ra_edges[0])
    safe_dec = np.where(finite_pointing, dec_flat, dec_edges[0])
    if is_regular:
        ix = np.floor((safe_ra - ra_edges[0]) / ra_step[0]).astype(np.int64)
        iy = np.floor((safe_dec - dec_edges[0]) / dec_step[0]).astype(np.int64)
    else:
        ix = np.digitize(safe_ra, ra_edges) - 1
        iy = np.digitize(safe_dec, dec_edges) - 1

    valid = (
        finite_pointing & (flags != 0)
        & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    )
    if not np.any(valid):
        empty = np.zeros((ny, nx), dtype=float)
        return empty, empty.copy(), empty.copy(), empty.copy(), empty.copy()

    valid_indices = np.flatnonzero(valid)
    flat_idx = iy[valid_indices] * nx + ix[valid_indices]
    valid_flags = flags[valid_indices]
    naive_values = naive_flat[valid] * valid_flags
    weighted_flags = valid_flags * det_w[valid_indices % n_dets]
    weighted_values = sig_flat[valid] * weighted_flags

    naive_data = np.bincount(
        flat_idx, weights=naive_values, minlength=nx * ny
    ).reshape(ny, nx)
    naive_hits = np.bincount(
        flat_idx, weights=valid_flags, minlength=nx * ny
    ).reshape(ny, nx)
    data = np.bincount(
        flat_idx, weights=weighted_values, minlength=nx * ny
    ).reshape(ny, nx)
    hits = np.bincount(
        flat_idx, weights=weighted_flags, minlength=nx * ny
    ).reshape(ny, nx)
    sumsq = np.bincount(
        flat_idx, weights=sig_flat[valid] * weighted_values,
        minlength=nx * ny
    ).reshape(ny, nx)
    return naive_data, naive_hits, data, hits, sumsq


'''
def bin_chunk(
        signal: np.ndarray, flag_mask: np.ndarray,
        ra: np.ndarray, dec: np.ndarray,
        ra_edges: np.ndarray, dec_edges: np.ndarray,
        weights: np.ndarray = None):
    """Bin one Chunk's signal into a 2-D pixel map.

    signal  : (n_samps, n_dets) cleaned signal
    flags_mask: (n_sampls, n_dets) mask of signal
    ra, dec : (n_samps, n_dets) per-detector pointing in degrees
    weights : (n_dets,) optional per-detector weights; defaults to uniform'

    Returns data, hits, and sumsq (weighted sum of signal**2), all shape (ny, nx).
    sumsq accumulates per-pixel variance (E[x^2] - E[x]^2) across chunks for a real RMS noise map.
    """

    # Initialize detector weights and sanitize flags against NaN/inf
    n_dets = signal.shape[1]
    det_w = np.ones(n_dets, dtype=float) if weights is None else np.asarray(weights, dtype=float)

    # Flatten multi-detector/time arrays into 1D for vector operations
    sig_orig = np.ascontiguousarray(signal).ravel()
    ra_flat = np.ascontiguousarray(ra).ravel()
    dec_flat = np.ascontiguousarray(dec).ravel()

    # Compute weights and apply to signal streams
    flags = np.nan_to_num(flag_mask, nan=0.0, posinf=0.0, neginf=0.0)
    w_flat = (flags * det_w[np.newaxis, :]).ravel()
    sig_flat = sig_orig * w_flat
    sig_sq_flat = sig_orig * sig_flat

    # Extract grid dimensions and origin coordinates
    ny = dec_edges.size - 1
    nx = ra_edges.size - 1
    ra0 = float(ra_edges[0])
    dec0 = float(dec_edges[0])

    ra_step = np.diff(ra_edges)
    dec_step = np.diff(dec_edges)

    # Fast Path: Use integer indexing and bincount for regular grids
    if ra_step.size and dec_step.size and np.allclose(ra_step, ra_step[0]) and np.allclose(dec_step, dec_step[0]):
        ra_bin = float(ra_step[0])
        dec_bin = float(dec_step[0])

        # Compute integer pixel indices
        ix = np.floor((ra_flat - ra0) / ra_bin).astype(np.int64, copy=False)
        iy = np.floor((dec_flat - dec0) / dec_bin).astype(np.int64, copy=False)
        
        # Validity mask: 
        # drops out-of-bounds points instead of clamping them to borders
        valid = (
            np.isfinite(ra_flat) & np.isfinite(dec_flat) & 
            (w_flat != 0.0) & 
            (ix >= 0) & (ix < nx) & 
            (iy >= 0) & (iy < ny)
        )
        
        if not np.any(valid):
            empty_map = np.zeros((ny, nx), dtype=float)
            return empty_map, empty_map.copy(), empty_map.copy()
        
        flat_idx = iy[valid] * nx + ix[valid]

        # Accumulate weighted maps using bincount
        data = np.bincount(flat_idx, weights=sig_flat[valid], minlength=nx * ny).reshape(ny, nx)
        hits = np.bincount(flat_idx, weights=w_flat[valid], minlength=nx * ny).reshape(ny, nx)
        sumsq = np.bincount(flat_idx, weights=sig_sq_flat[valid], minlength=nx * ny).reshape(ny, nx)
        return data, hits, sumsq

    # Fallback Path: Use histogram2d for irregular or non-uniform grids
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
'''