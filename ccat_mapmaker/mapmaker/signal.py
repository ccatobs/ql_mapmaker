# ============================================================================ #
# signal.py
#
# Converting raw detector data into calibrated frequency-shift timestreams.
#
# MKID detectors respond to light by shifting their resonant frequency. The
# readout records complex transmission S21 as I (real) and Q (imaginary).
# We convert (I, Q) to fractional frequency shift dF/F0, which is proportional
# to optical power.
#
# Not used for simulation data -- signal is already a processed quantity.
# ============================================================================ #

import numpy as np


def iq_to_df(I: np.ndarray, Q: np.ndarray,
             If: np.ndarray, Qf: np.ndarray, Ff: np.ndarray,
             i_f0: int = None) -> np.ndarray:
    """
    Convert I/Q timestreams to fractional frequency shift dF/F0 using the IQ
    angle method.

    As the resonant frequency shifts, the measured (I, Q) point travels around
    the resonance loop in the IQ plane. The angular position encodes the
    frequency shift, recovered by comparing to a calibration sweep.

    Steps:
      1. Estimate the centre of the IQ resonance circle from the sweep.
      2. Compute angle theta of each observed point around that centre.
      3. Compute angle theta_f for each calibration sweep point.
      4. Interpolate to find the frequency corresponding to each observed angle.

    I, Q   : (n_samps,) observed timestream, real and imaginary parts of S21
    If, Qf : (n_sweep_pts,) calibration sweep, real and imaginary parts
    Ff     : (n_sweep_pts,) frequency axis for the calibration sweep (Hz)
    i_f0   : index of resonant frequency in Ff; auto-detected as |S21| minimum if None

    Returns df : (n_samps,) fractional frequency shift dF/F0 (dimensionless)
    """
    if i_f0 is None:
        i_f0 = np.argmin(np.abs(If + 1j * Qf))

    # Centre of the IQ resonance circle (midpoint of sweep bounding box)
    cI = (If.max() + If.min()) / 2
    cQ = (Qf.max() + Qf.min()) / 2

    theta_f = np.arctan2(Qf - cQ, If - cI)
    theta   = np.arctan2(Q  - cQ, I  - cI)

    Ff0 = Ff - Ff[i_f0]  # shift frequency axis so F0 = 0

    # period=2*pi handles wrap-around of angles
    df = np.interp(theta, theta_f, Ff0, period=2 * np.pi)

    return df / Ff[i_f0]


def normalize_tod(tod_1d: np.ndarray, cal_lamp_tod_1d: np.ndarray) -> np.ndarray:
    """
    Normalise a detector timestream so the median is 0 and the cal lamp peak is 1.

    Removes DC offset and puts all detectors on the same scale regardless of
    individual sensitivity differences.

    tod_1d          : science observation timestream for one detector
    cal_lamp_tod_1d : cal lamp exposure timestream for the same detector

    Returns normalised timestream, or zeros if detector is dead or cal lamp
    didn't fire.
    """
    median_val = np.median(tod_1d)
    cal_peak   = np.max(cal_lamp_tod_1d)

    scale = cal_peak - median_val
    if scale == 0:
        return np.zeros_like(tod_1d)

    return (tod_1d - median_val) / scale
