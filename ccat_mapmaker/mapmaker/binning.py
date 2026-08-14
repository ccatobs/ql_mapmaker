# ============================================================================ #
# binning.py
#
# Sky map pixelization: bin detector timestreams into a 2-D map.
#
# For each time sample, we know where the detector was pointing and what signal
# it recorded. We divide the sky into a rectangular pixel grid and accumulate
# signal from all samples that fell in each pixel. The map value per pixel is
# the average of all signal samples that landed there, computed via
# numpy's histogram2d.
# ============================================================================ #

import numpy as np


# Map grid pattern adapted from Bonnie Slocombe, https://github.com/bonnieslocombe/g3_mapmaking, g3mapmaker.py, QuickMapMaker.__init__
# and Jonah Lee, https://github.com/jonahjlee/blasttng-to-g3, maps.py, MapBinner.__init__
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


# Adapted from Bonnie Slocombe, https://github.com/bonnieslocombe/g3_mapmaking, g3mapmaker.py, QuickMapMaker.Process
# and Jonah Lee, https://github.com/jonahjlee/blasttng-to-g3, maps.py, MapBinner.__call__
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


def bin_chunk(signal: np.ndarray, flag_mask: np.ndarray,
              ra: np.ndarray, dec: np.ndarray,
              ra_edges: np.ndarray, dec_edges: np.ndarray,
              weights: np.ndarray = None):
    """
    Bin one Chunk's signal into a 2-D pixel map.

    Flattens all detectors and time samples into a single histogram2d call.
    The caller accumulates (data, hits) across chunks to build the full map.

    signal  : (n_samps, n_dets) cleaned signal
    flags_mask: (n_sampls, n_dets) mask of signal
    ra, dec : (n_samps, n_dets) per-detector pointing in degrees
    weights : (n_dets,) optional per-detector weights; defaults to uniform
    Returns data, hits, and sumsq (weighted sum of signal**2), all shape (ny, nx).
    sumsq lets the caller accumulate per-pixel variance (E[x^2] - E[x]^2) across
    chunks for a real RMS noise map, instead of assuming unit variance per sample.
    """
    n_samps, n_dets = signal.shape
    w = np.ones(n_dets, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    flags = np.copy(flag_mask)
    flags[np.isnan(flags)] = 0
    flags_det_mean = np.mean(flags, axis = 0)
    flags_det_mean[np.isnan(flags_det_mean) == True] = 0
    flags_det_mean[flags_det_mean < 1] = 0
    w = w*flags_det_mean

    sig_flat    = (signal * w[np.newaxis, :]).ravel()
    sig_sq_flat = ((signal ** 2) * w[np.newaxis, :]).ravel()
    ra_flat     = ra.ravel()
    dec_flat    = dec.ravel()
    w_flat      = np.tile(w, n_samps)

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