# ============================================================================ #
# signal.py
#
# Converting raw detector data into calibrated frequency-shift timestreams.
#
# Code based on algorithm developed and refined by Max Chapman (https://github.com/freermax9-gif/CCAT-MKID-)
# ============================================================================ #

from typing import Optional
import numpy as np


# the hybrid method is a method that default uses a gradient method to estimate df from the I Q data. When the error is fraction of the sweep bandwidth beyond which the linear approximation is considered unreliable and the angle method is used instead (more accurate but slower). The threshold_frac parameter controls this threshold.

def iq_to_df_hybrid(I: np.ndarray, Q: np.ndarray,
                     If: np.ndarray, Qf: np.ndarray, Ff: np.ndarray,
                     i_f0: Optional[int] = None, threshold_frac: float = 0.05,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert I/Q timestreams to fractional frequency shift dF/F0, using a fast
    linear approximation near resonance and falling back to the slower,
    more accurate IQ angle method (iq_to_df) away from it.
    """
    if i_f0 is None:
        i_f0 = int(np.argmin(np.abs(If + 1j * Qf)))

    df = iq_to_df_gradient(I, Q, If, Qf, Ff, i_f0=i_f0)

    bandwidth = Ff.max() - Ff.min()
    threshold = threshold_frac * bandwidth / Ff[i_f0]
    used_fallback = np.abs(df) > threshold
    if used_fallback.any():
        df[used_fallback] = iq_to_df(I[used_fallback], Q[used_fallback], If, Qf, Ff, i_f0=i_f0)

    return df, used_fallback

def iq_to_df_gradient(I: np.ndarray, Q: np.ndarray,
             If: np.ndarray, Qf: np.ndarray, Ff: np.ndarray,
             i_f0: int = None) -> np.ndarray:
    """
    Convert I/Q timestreams to fractional frequency shift dF/F0 using a fast linear (tangent-line) approximation at the resonance point.
    """
    if i_f0 is None:
        i_f0 = int(np.argmin(np.abs(If + 1j * Qf)))

    i_grad = min(i_f0, len(If) - 2)
    dIf = np.diff(If)[i_grad] / 1e3
    dQf = np.diff(Qf)[i_grad] / 1e3
    denom = dIf ** 2 + dQf ** 2

    df = ((I-If[i_f0]) * dIf + (Q-Qf[i_f0]) * dQf) / denom / Ff[i_f0]

    return df

def iq_to_df(I: np.ndarray, Q: np.ndarray,
             If: np.ndarray, Qf: np.ndarray, Ff: np.ndarray,
             i_f0: int = None) -> np.ndarray:
    """
    Convert I/Q timestreams to fractional frequency shift dF/F0 using the IQ
    angle method.

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
    Removes DC offset and puts all detectors on the same scale regardless of individual sensitivity differences.
    """
    median_val = np.median(tod_1d)
    cal_peak   = np.max(cal_lamp_tod_1d)

    scale = cal_peak - median_val
    if scale == 0:
        return np.zeros_like(tod_1d)

    return (tod_1d - median_val) / scale
