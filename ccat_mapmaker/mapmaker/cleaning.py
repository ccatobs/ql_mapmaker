# ============================================================================ #
# cleaning.py
#
# James Burgoyne, jburgoyne@phas.ubc.ca
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT August 2026
# ============================================================================ #
"""
Timestream cleaning: remove artifacts from detector data.

Each cleaning step is a self-contained function registered in CLEAN_STEPS. clean_tod() runs whichever steps the pipeline config names, in order -- to add a new artifact-removal step (e.g. step correction), write one function with the standard (tod, flag_mask, sample_rate, **params) -> (tod, flags) signature (flags is a (n_samps, n_dets) int bitmask of newly-detected reasons, or None if the step doesn't detect anything e.g. highpass/notch are filters, not detectors), add it to CLEAN_STEPS, and reference it from config.toml's clean_steps list. No changes to clean_tod itself are needed.

Sources of contamination and where they're handled:
  Cosmic ray strikes   - sudden large spikes; removed here by interpolation
  Atmospheric noise    - slow low-frequency drift; removed here by high-pass filter
  Line pickup          - 50/60 Hz mains + harmonics; removed here by notch filter
  Correlated noise     - same across all detectors; handled by common_mode.py
  Unidentified narrowband contaminants - not removed, only flagged (see    find_psd_anomalies) for follow-up analysis. Motivated by AMKID (Reyes et al. 2026) finding an unexplained ~0.3 Hz line that persisted after magnetic shielding
"""


import numpy as np
from scipy.signal import medfilt
import scipy.fft as fft
import numba


# Bit values for chunk.flags (see reader.py's Chunk docstring for the
# bitmask convention: 0 = good, nonzero = OR of reasons). Add a new bit here
# per new detection reason; consumers that don't care which reason should
# still just test `flags != 0`.
FLAG_COSMIC_RAY = 1 << 0


# ============================================================================ #
# remove_cosmic_rays
# ============================================================================ #
def remove_cosmic_rays(tod_1d: np.ndarray, flag_mask: np.ndarray,
                       sigma: float = 3.5,
                       n: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect and interpolate over cosmic ray spikes in one detector's timestream.

    Flags samples where the step to the next sample exceeds sigma * std(tod),
    expands each flag by n samples on each side, then interpolates over flagged
    regions from surrounding clean data.

    Returns (tod, mask),  mask is a (n_samps,) bool array marking every
    sample the spike expansion covered, whether or not it could actually be
    interpolated (e.g. too few valid neighbours), so callers can record the
    detection even when the interpolation itself was skipped.
    """
    tod = tod_1d.copy()

    dtod = np.diff(tod*flag_mask, append=tod[-1])

    threshold    = np.nanstd((tod*flag_mask)) * sigma
    spike_indices = np.where(np.abs(dtod) > threshold)[0] + 1

    mask = np.zeros(len(tod), dtype=bool)
    if len(spike_indices) == 0:
        return tod, mask

    for idx in spike_indices:
        start = max(0, idx - n)
        end   = min(len(tod), idx + n + 1)
        mask[start:end] = True

    valid = np.where(~mask)[0]
    if len(valid) < 2:
        return tod, mask
    tod[mask] = np.interp(np.where(mask)[0], valid, tod[valid])

    return tod, mask


# ============================================================================ #
# highpass_filter
# ============================================================================ #
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


# ============================================================================ #
# detect_line_freq
# called in the notch filter function 
# ============================================================================ #
def detect_line_freq(tod: np.ndarray, sample_rate: float,
                     search_range_hz: tuple = (45.0, 65.0)) -> float:
    """
    Auto-detect the mains line frequency from the dominant narrow peak in
    search_range_hz (default spans both 50 Hz and 60 Hz mains).

    Uses the median power spectrum across all detectors, since line pickup is
    common-mode (affects every channel), which is better than picking a
    peak from any single noisy detector. Returns 0.0 if no peak is found
    (e.g. too few samples to resolve the search band).
    """
    n_samps = tod.shape[0]
    freqs   = fft.rfftfreq(n_samps, d=1.0 / sample_rate)
    band    = (freqs >= search_range_hz[0]) & (freqs <= search_range_hz[1])
    if not np.any(band):
        return 0.0

    power        = np.abs(fft.rfft(tod, axis=0)) ** 2
    median_power = np.median(power[band, :], axis=1)
    return float(freqs[band][np.argmax(median_power)])


# ============================================================================ #
# highpass_filter
# this method may not be useful for real data
# ============================================================================ #
def find_psd_anomalies(psd_avg: np.ndarray, psd_freqs: np.ndarray,
                       sigma: float = 5.0,
                       smooth_bins: int = 21,
                       skip_hz: float = 0.05) -> list:
    """
    Flag narrowband spectral features standing above the smooth 1/f + white
    continuum, for logging (not removal) so they can be looked into later 
    e.g. AMKID's unexplained ~0.3 Hz line (Reyes et al. 2026), which wasn't at
    a mains-related frequency and persisted even after magnetic shielding.

    Takes the median power spectrum across detectors (psd_avg, psd_freqs -
    the same arrays _first_pass already computes for white_noise_floor), and
    compares its log10 against a median-filtered local continuum: bins whose
    residual exceeds sigma * MAD stand out from the smooth background.
    Adjacent flagged bins are merged into one anomaly, reported at the bin of
    peak residual. Same log-space/MAD-based approach as wnf_exclude_sigma and
    Chapin et al. 2013 Section 3.1.6, the PSD's heavy-tailed shape makes a
    linear threshold unreliable.

    skip_hz excludes near-DC bins, which aren't a meaningful "narrowband"
    feature and would otherwise dominate the continuum fit.

    Caveat: resolution is set by psd_freqs' bin spacing (1/chunk_duration_s
    with the default _first_pass periodogram), a feature narrower than one
    bin, or below the lowest resolved frequency, won't be distinguishable.

    Returns a list of {"freq_hz", "power_ratio_db"} dicts, most significant
    first; empty if nothing stands out or there isn't enough spectrum to
    fit a continuum against.
    """
    good = psd_freqs >= skip_hz
    if good.sum() < smooth_bins:
        return []

    freqs      = psd_freqs[good]
    median_psd = np.median(psd_avg[good, :], axis=1)

    finite_psd = median_psd > 0
    if not np.any(finite_psd):
        return []
    log_psd = np.full_like(median_psd, -np.inf)
    log_psd[finite_psd] = np.log10(median_psd[finite_psd])

    # medfilt zero-pads at the boundary by default, which drags a log-power
    # continuum estimate artificially low right at the edges (log_psd is
    # always positive) and creates a spurious residual there. Edge-pad with
    # the boundary value instead, so the low- and high-frequency ends stay
    # usable, important since the low-frequency end (near the 1/f knee) is
    # exactly where a real contaminant like AMKID's ~0.3 Hz line would sit.
    kernel = smooth_bins if smooth_bins % 2 == 1 else smooth_bins + 1
    filled  = np.where(finite_psd, log_psd, np.median(log_psd[finite_psd]))
    pad     = kernel // 2
    padded  = np.pad(filled, pad, mode="edge")
    continuum = medfilt(padded, kernel_size=kernel)[pad:pad + len(filled)]
    residual  = log_psd - continuum

    finite = finite_psd & np.isfinite(residual)
    if not np.any(finite):
        return []

    mad = np.median(np.abs(residual[finite] - np.median(residual[finite])))
    if mad == 0:
        return []
    threshold = sigma * 1.4826 * mad

    flagged = finite & (residual > threshold)
    if not np.any(flagged):
        return []

    idx    = np.where(flagged)[0]
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)

    anomalies = []
    for g in groups:
        peak = g[np.argmax(residual[g])]
        anomalies.append({
            "freq_hz":        float(freqs[peak]),
            "power_ratio_db": float(10.0 * residual[peak]),  # log10(power) residual -> dB
        })

    anomalies.sort(key=lambda a: -a["power_ratio_db"])
    return anomalies


# ============================================================================ #
# notch_filter
# ============================================================================ #
def notch_filter(tod: np.ndarray, flag_mask: np.ndarray, sample_rate: float,
                 freq_hz: float = 0.0,
                 n_harmonics: int = 5,
                 bandwidth_hz: float = 1.0) -> np.ndarray:
    """
    Remove line-frequency (mains) pickup and its harmonics via FFT notch
    filtering on the full (n_samps, n_dets) array.

    freq_hz <= 0 (default) auto-detects the fundamental per call from the
    dominant peak in the 45-65 Hz band, so the same config works whether a
    site's mains run at 50 Hz or 60 Hz. Set freq_hz explicitly (e.g. 60.0) to
    force a fixed frequency instead. n_harmonics counts the fundamental as
    harmonic 1; harmonics at or above the Nyquist frequency are skipped.
    """
    n_samps = tod.shape[0]

    freq = freq_hz if freq_hz > 0.0 else detect_line_freq(tod, sample_rate)
    if freq <= 0.0:
        return tod

    ### Cannot handle Nan Values cleanly; NaN values are instead set to 0 ###
    flags = flag_mask
    flags[np.isnan(flag_mask)] = 0
    tod_fft = fft.rfft(tod*flags, axis=0)
    freqs   = fft.rfftfreq(n_samps, d=1.0 / sample_rate)

    nyquist = sample_rate / 2.0
    for h in range(1, n_harmonics + 1):
        f_h = freq * h
        if f_h >= nyquist:
            break
        tod_fft[np.abs(freqs - f_h) <= bandwidth_hz / 2.0, :] = 0.0

    return fft.irfft(tod_fft, n=n_samps, axis=0)


# ============================================================================ #
# _step_cosmic_rays
# based on Chapin_2013 where they apply a step correction
# in progress, and not tested
# ============================================================================ #
def _step_cosmic_rays(tod: np.ndarray, flag_mask: np.ndarray, sample_rate: float,
                      sigma: float = 3.5, n: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """CLEAN_STEPS wrapper: apply remove_cosmic_rays per detector."""
    tod_clean = tod.copy()
    new_flags = np.zeros(tod.shape, dtype=int)
    for i in range(tod_clean.shape[1]):
        tod_clean[:, i], mask = remove_cosmic_rays(tod_clean[:, i], flag_mask[:, i],
                                                    sigma=sigma, n=n)
        new_flags[mask, i] = FLAG_COSMIC_RAY
    return tod_clean, new_flags


# ============================================================================ #
# _step_highpass
# ============================================================================ #
def _step_highpass(tod: np.ndarray, flag_mask: np.ndarray, sample_rate: float,
                   cutoff_hz: float = 0.5) -> tuple[np.ndarray, None]:
    """CLEAN_STEPS wrapper: brick-wall high-pass filter across the full array."""
    if cutoff_hz <= 0.0:
        return tod, None

    n_samps = tod.shape[0]
    ### Cannot handle Nan Values cleanly; NaN values are instead set to 0 ###
    flags = flag_mask
    flags[np.isnan(flag_mask)] = 0
    tod_fft = fft.rfft(tod*flags, axis=0)
    freqs   = fft.rfftfreq(n_samps, d=1.0 / sample_rate)
    tod_fft[freqs < cutoff_hz, :] = 0.0
    return fft.irfft(tod_fft, n=n_samps, axis=0), None


# ============================================================================ #
# _step_notch
# ============================================================================ #
def _step_notch(tod: np.ndarray, flag_mask: np.ndarray, sample_rate: float,
                freq_hz: float = 0.0, n_harmonics: int = 5,
                bandwidth_hz: float = 1.0) -> tuple[np.ndarray, None]:
    """CLEAN_STEPS wrapper: line-frequency notch filter, see notch_filter()."""
    return notch_filter(tod, flag_mask, sample_rate,
                        freq_hz=freq_hz, n_harmonics=n_harmonics,
                        bandwidth_hz=bandwidth_hz), None


CLEAN_STEPS = {
    "cosmic_rays": _step_cosmic_rays,
    "highpass":    _step_highpass,
    "notch":       _step_notch,
}


# ============================================================================ #
# clean_tod
# ============================================================================ #
def clean_tod(tod: np.ndarray, flag_mask: np.ndarray, sample_rate: float,
              steps: list, step_params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the named cleaning steps to a full (n_samps, n_dets) array, in order.

    steps is an ordered list of names from CLEAN_STEPS (e.g.
    ["cosmic_rays", "highpass", "notch"]); step_params optionally maps each
    name to a dict of keyword overrides for that step. Unlisted steps are
    skipped entirely.

    Per-detector baseline removal is handled before this call. Correlated
    noise removal is handled by common_mode.py.

    Returns (tod_clean, new_flags). new_flags is a (n_samps, n_dets) int
    bitmask of reasons detected by the steps that ran (e.g. FLAG_COSMIC_RAY),
    OR'd across all steps,  0 everywhere if none of the ran steps detect
    anything (e.g. highpass/notch, which are filters, not detectors). This is
    NOT the same array as flag_mask that was passed in, flag_mask isn't
    modified, so the caller decides where the new flags get used (e.g. only
    for map binning, not fed back into this same cleaning pass).
    """
    step_params = step_params or {}
    tod_clean = tod.copy()
    new_flags = np.zeros(tod.shape, dtype=int)

    for name in steps:
        if name not in CLEAN_STEPS:
            raise ValueError(f"Unknown cleaning step {name!r}; available: {sorted(CLEAN_STEPS)}")
        tod_clean, step_flags = CLEAN_STEPS[name](tod_clean, flag_mask, sample_rate,
                                                   **step_params.get(name, {}))
        if step_flags is not None:
            new_flags |= step_flags

    return tod_clean, new_flags
