# ============================================================================ #
# signal.py
# IQ -> df functions.
#
#
# jburgoyne@phas.ubc.ca
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT 2026
# ============================================================================ #

# from typing import Optional
import numpy as np


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
# ============================================================================ #
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
# ============================================================================ #
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
# ============================================================================ #
def find_ftone_median(I, Q, If, Qf, Ff):
    """Find the probe tone by taking median of observation data.
    This is not perfect, but not terrible in absence of actual probe tone info.
    """

    Ipt = np.median(I)
    Qpt = np.median(Q)

    sq_dist = (If - Ipt)**2 + (Qf - Qpt)**2

    i_ft = np.argmin(sq_dist)

    return If[i_ft], Qf[i_ft], Ff[i_ft], i_ft


# ============================================================================ #
# find_i_ftone
# ============================================================================ #
def find_i_ftone(Ff, f_tone):
    """Find the index in the frequency sweep that the probe tone is closest to.
    """
    
    return np.abs(Ff - f_tone).argmin()


# ============================================================================ #
# iq_to_df_gradient
# ============================================================================ #
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
# ============================================================================ #
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
# ============================================================================ #
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




# ============================================================================ #
# normalize_tod
# ============================================================================ #
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
