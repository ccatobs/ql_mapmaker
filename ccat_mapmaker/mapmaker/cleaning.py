# ============================================================================ #
# cleaning.py
#
# Timestream cleaning: remove non-astronomical artifacts from detector data.
#
# Sources of contamination and where they're handled:
#   Cosmic ray strikes   - sudden large spikes; removed here by interpolation
#   Atmospheric noise    - slow low-frequency drift; removed here by high-pass filter
#   Correlated noise     - same across all detectors; handled by common_mode.py
#   60 Hz pickup         - power line interference; notch filter (not yet implemented)
# ============================================================================ #

import numpy as np
import scipy.fft as fft


#step correction - should be modular enough that if necessary we can implement this

def remove_cosmic_rays(tod_1d: np.ndarray, flag_mask: np.ndarray,
                       sigma: float = 3.5,
                       n: int = 2) -> np.ndarray:
    """
    Detect and interpolate over cosmic ray spikes in one detector's timestream.

    Flags samples where the step to the next sample exceeds sigma * std(tod),
    expands each flag by n samples on each side, then interpolates over flagged
    regions from surrounding clean data.
    """
    tod = tod_1d.copy()

    dtod = np.diff(tod*flag_mask, append=tod[-1])

    threshold    = np.nanstd((tod*flag_mask)) * sigma
    spike_indices = np.where(np.abs(dtod) > threshold)[0] + 1

    if len(spike_indices) == 0:
        return tod

    mask = np.zeros(len(tod), dtype=bool)
    for idx in spike_indices:
        start = max(0, idx - n)
        end   = min(len(tod), idx + n + 1)
        mask[start:end] = True

    valid = np.where(~mask)[0]
    if len(valid) < 2:
        return tod
    tod[mask] = np.interp(np.where(mask)[0], valid, tod[valid])

    return tod


def highpass_filter(tod_1d: np.ndarray,
                    sample_rate: float,
                    cutoff_hz: float) -> np.ndarray:
    """
    Remove slow variations below cutoff_hz via brick-wall FFT high-pass filter.
    Set cutoff_hz <= 0 to disable.
    """
    if cutoff_hz <= 0.0:
        return tod_1d

    tod_fft = fft.rfft(tod_1d)
    freqs   = fft.rfftfreq(len(tod_1d), d=1.0 / sample_rate)
    tod_fft[freqs < cutoff_hz] = 0.0
    return fft.irfft(tod_fft, n=len(tod_1d))


def clean_tod(tod: np.ndarray, flag_mask: np.ndarray,
              sample_rate: float,
              cosmic_rays: bool = True,
              highpass_hz: float = 0.5) -> np.ndarray:
    """
    Apply all enabled cleaning steps to a full (n_samps, n_dets) array.

    Steps in order:
        1. Cosmic ray removal  (if cosmic_rays=True)
        2. High-pass filter    (if highpass_hz > 0)

    Per-detector baseline removal is handled before this call. Correlated
    noise removal is handled by common_mode.py.
    """
    tod_clean = tod.copy()
    n_samps, n_dets = tod_clean.shape

    if cosmic_rays:
        for i in range(n_dets):
            tod_clean[:, i] = remove_cosmic_rays(tod_clean[:, i], flag_mask[:, i])

    if highpass_hz > 0.0:
        ### Cannot handle Nan Values cleanly; NaN values are instead set to 0 ###
        flags = flag_mask
        flags[np.isnan(flag_mask)] = 0
        tod_fft = fft.rfft(tod_clean*flags, axis=0)
        freqs   = fft.rfftfreq(n_samps, d=1.0 / sample_rate)
        tod_fft[freqs < highpass_hz, :] = 0.0
        tod_clean = fft.irfft(tod_fft, n=n_samps, axis=0)

    return tod_clean
