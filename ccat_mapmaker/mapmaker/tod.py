# ============================================================================ #
# tod.py
#
# James Burgoyne, jamesrburgoyne@icloud.com
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
#
# CCAT 2026
# ============================================================================ #

import numpy as np
from scipy.signal import medfilt
import scipy.fft as fft


# Bit values for chunk.flags (see reader.py's Chunk docstring for the
# bitmask convention: 0 = good, nonzero = OR of reasons). Add a new bit here
# per new detection reason; consumers that don't care which reason should
# still just test `flags != 0`.
FLAG_COSMIC_RAY = 1 << 0




# ============================================================================ #
# == CLEANING ==
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


# ============================================================================ #
# findWhiteNoiseFloor
def findWhiteNoiseFloor(psd_avg, psd_freqs, sample_rate):
    """Estimate the white noise floor from an averaged Power Spectral Density (PSD) matrix.

    psd_avg (2D array of floats): PSD values with shape (frequency_bins, channels).
    psd_freqs (array): Frequency values corresponding to the rows of psd_avg.
    sample_rate (float) Sampling rate of the data in Hertz.

    Returns:
        (array or None): 1D array containing the median white noise floor for each channel, or None if psd_avg is None.
    """

    if psd_avg is None:
        return None

    nyquist = sample_rate / 2

    # Isolate the high-frequency band (40% to 90% of Nyquist) to avoid 
    # low-frequency 1/f noise components and high-frequency edge roll-off.
    white_band = (psd_freqs > 0.4 * nyquist) & (psd_freqs < 0.9 * nyquist)

    # Compute the robust median across the frequency band for each channel
    white_noise_floor = np.median(psd_avg[white_band, :], axis=0)

    return white_noise_floor


# ============================================================================ #
# findNoise
def findNoise(sum, sum_sq, count):
    """Estimate noise as standard deviation.

    sum (array of floats): Sum of samples.
    sum_sq (array of floats): Sum of squared samples.
    count (arrays of ints): Number of samples.

    Returns:
        std (array of floats): Estimated standard deviation (sigma).
    """
    
    var = (sum_sq/count) - (sum/count)**2
    noise = np.sqrt(np.maximum(var, 0))

    return noise


# ============================================================================ #
# _findPsdAnomalies
def _findPsdAnomalies(psd_avg: np.ndarray, psd_freqs: np.ndarray,
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
# _checkPsdAnomalies
def _checkPsdAnomalies(
        psd_avg: np.ndarray, psd_freqs: np.ndarray, 
        pipe_cfg: dict
        ) -> list:
    """
    """

    psd_anomalies = []
    if psd_avg is not None:
        anomaly_sigma = pipe_cfg.get("psd_anomaly_sigma", 5.0)
        if anomaly_sigma > 0:
            psd_anomalies = _findPsdAnomalies(
                psd_avg, psd_freqs, sigma=anomaly_sigma)
            if psd_anomalies:
                summary = ", ".join(f"{a['freq_hz']:.2f} Hz ({a['power_ratio_db']:.1f} dB)"
                                    for a in psd_anomalies[:5])
                print(f"  PSD anomalies flagged: {len(psd_anomalies)} ({summary}"
                      f"{', ...' if len(psd_anomalies) > 5 else ''})")
    return psd_anomalies


# ============================================================================ #
# _findNoiseExcludes
def _findNoiseExcludes(
        det_noise: np.ndarray, noise_thresh: float
        ) -> list[int]:
    """Identify detectors whose noise exceeds a multiple of the median noise level.

    Args:
        det_noise: Array of noise values for the detectors.
        noise_thresh: Multiplier threshold applied to the median noise level.

    Returns:
        Indices of detectors exceeding the noise threshold as a Python list.
    """
    
    # Establish the baseline median noise across the array
    det_noise_med = np.median(det_noise)
    
    # Flag detectors exceeding the threshold multiplier above the median
    noise_bad_i = np.flatnonzero(det_noise > noise_thresh * det_noise_med)
    
    return noise_bad_i.tolist()


# ============================================================================ #
# _findWnfExcludes
def _findWnfExcludes(
        wnf:np.ndarray, wnf_sigma:float
        ) -> tuple[list[int], float]:
    """Identify invalid or statistically anomalous detectors based on white noise floor.

    Computes robust statistics in log space to determine an upper noise floor
    cutoff, flagging non-positive/non-finite values and outliers exceeding 
    the specified sigma threshold.

    Args:
        wnf: Array of white noise floor values for resonators.
        wnf_sigma: Number of standard deviations above the log-space median 
            to use as the rejection threshold.

    Returns:
        A tuple containing:
            - Indices of bad or invalid resonators as a Python list.
            - The absolute linear-space noise floor cutoff value.
    """

    # Identify physically valid, finite, and positive noise floor entries
    valid_i = np.isfinite(wnf) & (wnf > 0)
    
    # Compute log-space statistics to handle orders-of-magnitude spread
    valid_log = np.log10(wnf[valid_i])
    valid_log_med = np.median(valid_log)
    valid_log_std = np.std(valid_log)
    
    # Project the sigma threshold back to linear space
    wnf_cutoff = 10**(valid_log_med + wnf_sigma * valid_log_std)
    
    # Flag invalid entries (~valid_i) and upper outliers in a single pass
    wnf_bad_i = np.flatnonzero(~valid_i | (wnf > wnf_cutoff))
    
    return wnf_bad_i.tolist(), wnf_cutoff


# ============================================================================ #
# excludedDetectors
def excludedDetectors(
    all_kids: list, det_noise: np.ndarray,
    white_noise_floor: np.ndarray | None, pipe_cfg: dict,
    ) -> tuple[np.ndarray, np.ndarray, set[int], list, list, float | None]:
    """Filters detectors based on manual config, noise thresholds, and white noise floor outliers.
    """

    n_total = len(all_kids)
    all_i = np.arange(n_total)
    kids_i = {k: i for i, k in enumerate(all_kids)}
    exclude_set = set()

    # Manual exclusions by name
    manual_names = pipe_cfg.get("exclude_detectors", [])
    exclude_set.update(kids_i[e] for e in manual_names if e in kids_i)

    # Manual exclusions by id
    manual_indices = pipe_cfg.get("exclude_detector_indices", [])
    exclude_set.update(i for i in manual_indices if 0 <= i < n_total)

    # Automatic noise-based exclusion
    noise_thresh = pipe_cfg.get("auto_exclude_threshold", 5.0)
    noise_excluded = []
    if noise_thresh > 0 and det_noise is not None and det_noise.size > 0:
        noise_bad_i = _findNoiseExcludes(det_noise, noise_thresh)
        noise_excluded = [all_kids[i] for i in noise_bad_i]
        exclude_set.update(noise_bad_i.tolist())

    # White Noise Floor (WNF) outlier exclusion
    wnf_sigma = pipe_cfg.get("wnf_exclude_sigma", 3.0)
    wnf_excluded = []
    wnf_cutoff = None
    if wnf_sigma > 0 and white_noise_floor is not None:
        wnf_bad_i, wnf_cutoff = _findWnfExcludes(white_noise_floor, wnf_sigma)
        wnf_excluded = [all_kids[i] for i in wnf_bad_i]
        exclude_set.update(wnf_bad_i)

    keep_idx = np.setdiff1d(all_i, exclude_set, assume_unique=True)

    print(f"Using {len(keep_idx)}/{n_total} detectors")

    return {
        'keep_idx':keep_idx,
        'exclude_set':exclude_set,
        'noise_excluded':noise_excluded,
        'wnf_excluded':wnf_excluded,
        'wnf_cutoff':wnf_cutoff,
    }


# ============================================================================ #
# findWeights
def findWeights(
        white_noise_floor:np.ndarray, n_total:int, pipe_cfg:dict
        ) -> np.ndarray:
    """Compute inverse-variance weights from a channel white noise floor.

    white_noise_floor (array of floats): White noise floor of each detector.
    n_total (int): Total number of detectors.
    pipe_cfg (dict): Pipeline configuration dictionary. 
        'weight_cap_factor' (float): Multiplier applied to the median 
            weight to cap extreme outliers (default: 5.0).

    Returns:
        (array of floats): Detector weights of length `n_total`. Non-finite or 
            non-positive entries are imputed with the median valid weight.
    """

    weight_cap_factor = pipe_cfg.get("weight_cap_factor", 5.0)

    det_weights = np.ones(n_total)
    if weight_cap_factor > 0 and white_noise_floor is not None:

        finite_wnf = np.isfinite(white_noise_floor) & (white_noise_floor > 0)
        if np.any(finite_wnf):

            # Inverse-variance weighting based on white noise floor power
            det_weights[finite_wnf] = 1.0 / white_noise_floor[finite_wnf]
            
            # Cap extreme weights
            # prevents near-zero noise channels from dominating
            median_w = np.median(det_weights[finite_wnf])
            det_weights = np.clip(det_weights, 0, weight_cap_factor * median_w)
            det_weights[~finite_wnf] = median_w

    return det_weights


# ============================================================================ #
# remove_cosmic_rays
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
# notch_filter
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


# ============================================================================ #
# normalize_tod
def normalize_tod(tod_1d: np.ndarray, cal_lamp_tod_1d: np.ndarray) -> np.ndarray:
    """
    Normalise a detector timestream so the median is 0 and the cal lamp peak is 1.
    Removes DC offset and puts all detectors on the same scale regardless of individual sensitivity differences.
    """
    # TODO: This should NOT be used except with BLAST-TNG data!
    median_val = np.median(tod_1d)
    cal_peak   = np.max(cal_lamp_tod_1d)

    scale = cal_peak - median_val
    if scale == 0:
        return np.zeros_like(tod_1d)

    return (tod_1d - median_val) / scale




# ============================================================================ #
# == COMMON MODE
# ============================================================================ #


# ============================================================================ #
# _mean_over_valid
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
def estimate_common_mode(tod: np.ndarray, flag_mask: np.ndarray) -> np.ndarray:
    """Mean across detectors at each time sample : naive atmosphere estimate."""
    # TODO: The detectors are pointing at different parts of the sky
    # so this only removes signal common at one time.
    # Elevation component of atmosphere not perfectly removed with this.
    # Can we remove it some other way?
    return _mean_over_valid(tod, flag_mask)


# ============================================================================ #
# subtract_common_mode
def subtract_common_mode(tod: np.ndarray,
                         common_mode: np.ndarray) -> np.ndarray:
    """Subtract the common-mode estimate from every detector's timestream."""
    return tod - common_mode[:, np.newaxis]


# ============================================================================ #
# lookup_map_signal
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




# ============================================================================ #
# == SWEEP ==
# ============================================================================ #

"""
Note: These techniques are based on a static tuning sweep.
    We are assuming that 'shifting' the sweep under the probe tone 
    is equal to what is actually happening: 
    The resonance is shifting in frequency and quality factor.
    These are obviously not actually equal things.
    The hope is they are equal enough.
    More exploration of this is suggested.
"""

"""
Note: Resonance is NOT necessarily where the tone was placed!
"""


# ============================================================================ #
# find_i_min
def find_i_min(If, Qf):
    """Find |S_21| min index in target sweep.
    This is very fast, but not accurate to identify resonance unless 
    using pre-calibrated sweep.

    If (np.array of floats): Tuning (target) sweep I components.
    Qf (np.array of floats): Tuning (target) sweep Q components.

    The magnitude minimum occurs where the distance from the origin (0,0) to 
    the resonance curve in the IQ plane is smallest. If the resonance circle 
    is rotated or shifted away from the origin due to impedance mismatches or 
    asymmetric coupling (often modeled by a complex coupling quality factor Qc), 
    the geometric point closest to the origin shifts away from fr. 
    Consequently, min(∣S21∣) rarely falls precisely on fr in practical devices.
    """

    # min in |S_21| (with cable delay not removed here)
    i_f0 = int(np.argmin(np.abs(If + 1j*Qf)))

    # If the min hits the last index-
    # we can't use that index for diff so shift one index lower.
    i_f0 = min(i_f0, len(If) - 2)

    return i_f0


# ============================================================================ #
# find_i_dvmax
def find_i_dvmax(If, Qf):
    """Find resonance index via maximum derivative in IQ space.
    
    If (np.array of floats): Tuning (target) sweep I components.
    Qf (np.array of floats): Tuning (target) sweep Q components.

    This marker corresponds strictly to the true natural resonance frequency. 
    In the complex IQ plane, the response forms a resonance circle. 
    The point where the trajectory travels fastest along the circle per unit 
    frequency is the point of maximum energy storage and quality factor 
    dissipation, making it invariant to foreign phase offsets or linear 
    transmission baselines.
    """
    
    z = If + 1j*Qf
    # Calculate discrete distance between adjacent sweep points
    dz = np.abs(np.diff(z))
    
    # Peak velocity (occurs at resonance)
    i_f0 = int(np.argmax(dz))
    
    # Cap upper index for safety
    return min(i_f0, len(If) - 2)


# ============================================================================ #
# find_ftone_median
def find_ftone_median(I, Q, If, Qf, Ff):
    """Find the probe tone by taking median of observation data.
    This is not perfect, but not terrible in absence of actual probe tone info.
    Note: This will not work well with time chunking.
    """

    Ipt = np.median(I)
    Qpt = np.median(Q)

    sq_dist = (If - Ipt)**2 + (Qf - Qpt)**2

    i_ft = np.argmin(sq_dist)

    return If[i_ft], Qf[i_ft], Ff[i_ft], i_ft


# ============================================================================ #
# find_i_ftone
def find_i_ftone(Ff, f_tone):
    """Find the index in the frequency sweep that the probe tone is closest to.
    """
    
    return np.abs(Ff - f_tone).argmin()



# ============================================================================ #
# == IQ TO DF ==
# ============================================================================ #


# ============================================================================ #
# iq_to_df_gradient
def iq_to_df_gradient(I, Q, If, Qf, Ff, f_tone=None, i_ft=None):
    """Convert I/Q timestreams to fractional frequency shift dF/F0 
    using a fast linear (tangent-line) approximation at the resonance point.

    I (np.array of floats):  Observation timestream I components.
    Q (np.array of floats):  Observation timestream Q components.
    If (np.array of floats): Tuning (target) sweep I components.
    Qf (np.array of floats): Tuning (target) sweep Q components.
    Ff (np.array of floats): Tuning (target) sweep frequency steps.
    f_tone (float):          Probe tone frequency.
    i_ft (int):              Probe tone index.
    """

    # Find the resonance index.
    # Maybe don't care about this!
    # i_f0 = find_i_dvmax(If, Qf)

    # Find the probe tone and/or index.
    # This sets baseline operating point that df is relative to.
    if i_ft is None:
        if f_tone is None:
            _, _, f_tone, i_ft = find_ftone_median(I, Q, If, Qf, Ff)
        else:
            i_ft = find_i_ftone(Ff, f_tone)
    # STORE: This could be a stored intermediary product.

    # Frequency sweep step at probe tone.
    dF = Ff[i_ft + 1] - Ff[i_ft]

    # Compute the local tangent vector dZ/dF = (dI/dF, dQ/dF).
    # This represents the velocity and direction of the IQ point 
    # along the resonance curve.
    dIf_dF = (If[i_ft + 1] - If[i_ft]) / dF # V/Hz
    dQf_dF = (Qf[i_ft + 1] - Qf[i_ft]) / dF

    # Squared magnitude of the local gradient vector ||dZ/dF||^2.
    # Used for normalizing the scalar projection onto the tangent line.
    denom = dIf_dF**2 + dQf_dF**2

    # Project the observed displacement vector delta_Z = (I - I0, Q - Q0) 
    # onto the tangent direction using the dot product: 
    # delta_f = (delta_Z . dZ/dF) / ||dZ/dF||^2.
    # This converts spatial displacement in the IQ plane 
    # to an equivalent frequency shift delta_f.
    df = ((I - If[i_ft])*dIf_dF + (Q - Qf[i_ft])*dQf_dF)/denom

    # Normalize by the resonant frequency F0 
    # to yield dimensionless fractional frequency shift (dF/F0).
    # This helps normalize responses across the array.
    return df / Ff[i_ft]


# ============================================================================ #
# iq_to_df_angle
def iq_to_df_angle(I, Q, If, Qf, Ff, f_tone=None, i_ft=None):
    """Convert I/Q timestreams to fractional frequency shift dF/F0 
    using a slow IQ angle method.

    I (np.array of floats):  Observation timestream I components.
    Q (np.array of floats):  Observation timestream Q components.
    If (np.array of floats): Tuning (target) sweep I components.
    Qf (np.array of floats): Tuning (target) sweep Q components.
    Ff (np.array of floats): Tuning (target) sweep frequency steps.
    f_tone (float):          Probe tone frequency.
    i_ft (int):              Probe tone index.
    """

    # Find the probe tone and/or index.
    # This sets baseline operating point that df is relative to.
    if i_ft is None:
        if f_tone is None:
            _, _, f_tone, i_ft = find_ftone_median(I, Q, If, Qf, Ff)
        else:
            i_ft = find_i_ftone(Ff, f_tone)
    # STORE: This could be a stored intermediary product.

    # Estimate the resonance circle center (cI, cQ) 
    # using the bounding box midpoint.
    # Translating the origin to the circle center transforms IQ trajectories 
    # into pure phase angles around the loop.
    cI = (If.max() + If.min()) / 2
    cQ = (Qf.max() + Qf.min()) / 2
    # STORE: The center could be more robustly fit if the results are stored
    # rather than recomputed for every chunk.

    # Map target sweep and observation timestream vectors to phase angles [rad] 
    # relative to the circle center. 
    # This decouples frequency-driven phase rotation from amplitude changes.
    theta_f = np.unwrap(np.arctan2(Qf - cQ, If - cI)) # STORE
    theta   = np.unwrap(np.arctan2(Q  - cQ, I  - cI))

    # Express frequency relative to probe tone (F0 = 0 Hz) to create a direct 
    # mapping from loop phase angle theta to frequency offset delta_f.
    dFf = Ff - Ff[i_ft]

    # Interpolate observed phase angles against the tuning sweep frequencies 
    # to reconstruct absolute frequency shifts df [Hz].
    # Setting period=2*pi handles phase wrap-around across the -pi/pi boundary.
    # Note: np.interp expects theta_f to be monotonically increasing.
    df = np.interp(theta, theta_f, dFf, period=2*np.pi)
    # STORE: This can NOT be stored.
    # However, we could attempt a different method that can be stored.
    # For example, if we can create a function of theta.

    # Normalize by the resonant frequency F0 
    # to yield dimensionless fractional frequency shift (dF/F0).
    # This helps normalize responses across the array.
    return df / Ff[i_ft]


# ============================================================================ #
# iq_to_df_hybrid
def iq_to_df_hybrid(I, Q, If, Qf, Ff, f_tone=None, dF_tol=4):
    """
    Convert I/Q timestreams to fractional frequency shift dF/F0, using a fast
    linear approximation near resonance and falling back to the slower,
    more accurate IQ angle method away from it.

    I (np.array of floats):  Observation timestream I components.
    Q (np.array of floats):  Observation timestream Q components.
    If (np.array of floats): Tuning (target) sweep I components.
    Qf (np.array of floats): Tuning (target) sweep Q components.
    Ff (np.array of floats): Tuning (target) sweep frequency steps.
    f_tone (float):          Probe tone frequency.
    dF_tol (float):          Frequency sweep steps away from resonance
                             to switch from grad to angle method.
    """

    # Find the probe tone and/or index.
    # This sets baseline operating point that df is relative to.
    if f_tone is None:
        _, _, f_tone, i_ft = find_ftone_median(I, Q, If, Qf, Ff)
    else:
        i_ft = find_i_ftone(Ff, f_tone)
    
    # First-pass: evaluate all samples using the fast gradient method.
    # This is computationally cheap (vectorized dot product) and accurate for 
    # small IQ perturbations near resonance where local curvature is negligible.
    df_f = iq_to_df_gradient(I, Q, If, Qf, Ff, f_tone)

    # Local frequency resolution of the sweep around probe tone [Hz/step].
    dF = Ff[i_ft + 1] - Ff[i_ft]

    # Flag samples where linear extrapolation exceeds the small-signal regime.
    # Converts df/f0 back to absolute frequency shift df [Hz] 
    # and checks if the excursion exceeds the tolerance window (dF_tol * dF). 
    # Large excursions suffer from tangent-line projection errors 
    # due to IQ loop curvature.
    i_redo = np.abs(df_f*Ff[i_ft]) > (dF*dF_tol)

    # Recalculate flagged samples with angle method.
    if i_redo.any():
        df_f[i_redo] = iq_to_df_angle(I[i_redo], Q[i_redo], If, Qf, Ff, f_tone)
    
    return df_f, i_redo