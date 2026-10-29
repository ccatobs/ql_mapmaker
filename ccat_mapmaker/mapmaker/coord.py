
# ============================================================================ #
# coord.py
#
# James Burgoyne, jamesrburgoyne@icloud.com
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT 2026
# ============================================================================ #

from typing import Optional

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, get_body
import astropy.units as u

import numpy as np
from scipy.spatial.transform import Rotation


# FYST site on Cerro Chajnantor for simulations and when there is real data
# TODO: move to config
FYST_SITE = EarthLocation(lat=-22.98592 * u.deg, lon=-67.74028 * u.deg, height=5610 * u.m)

_Z_AXIS = np.array([0.0, 0.0, 1.0])


# ============================================================================ #
# == TARGET ==
# ============================================================================ #




# ============================================================================ #
# isSolarSystemBody
def isSolarSystemBody(name: str) -> bool:

    _SOLAR_SYSTEM_BODIES = {
        "sun", "moon", "mercury", "venus", "mars",
        "jupiter", "saturn", "uranus", "neptune",
    }

    return name.strip().lower() in _SOLAR_SYSTEM_BODIES


# ============================================================================ #
# resolveTarget
def resolveTarget(name: str, obs_time_unix: Optional[float] = None,
                  site: Optional[EarthLocation] = None
                  ) -> tuple[float, float]:
    """Resolve a target name to (ra_deg, dec_deg).
    """

    if isSolarSystemBody(name):
        if obs_time_unix is None or site is None:
            raise ValueError(
                f"'{name}' is a solar-system body, obs_time_unix and site "
                f"are both required to look up its position."
            )
        t    = Time(obs_time_unix, format="unix", scale="utc")
        body = get_body(name.strip().lower(), t, site)
        return float(body.ra.deg), float(body.dec.deg)

    coord = SkyCoord.from_name(name)
    return float(coord.ra.deg), float(coord.dec.deg)


# ============================================================================ #
# resolveSite
def resolveSite(
        cfg:dict, target_name:str="target"
        ) -> EarthLocation | None:
    """Resolve the observing site (static attribute or telemetry-based).
    """

    map_cfg = cfg.get("map", {})
    
    # Look up static sites (e.g., "fyst" -> target.FYST_SITE)
    site_name = map_cfg.get("site", "fyst")
    site = globals().get(f"{site_name.upper()}_SITE")
    
    return site


# ============================================================================ #
# resolveMapCenter
def resolveMapCenter(
        cfg:dict, obs_info:dict, 
        ra0_auto:float, dec0_auto:float
        ) -> tuple[float, float, str]:
    """Resolve map center coordinates from config, target lookup, or auto-boresight.
    """

    map_cfg = cfg.get('map', {})

    # Explicitly pinned coordinates in config take precedence
    if 'ra0_deg' in map_cfg and 'dec0_deg' in map_cfg:
        ra0_deg, dec0_deg = map_cfg['ra0_deg'], map_cfg['dec0_deg']
        centre_source = 'config'

    # Named target resolution
    elif target_name := map_cfg.get('target'):

        site = resolveSite(cfg, target_name=target_name)
        
        # Solar system bodies require topocentric corrections (parallax)
        if isSolarSystemBody(target_name):
            if site is None:
                site_label = map_cfg.get("site", "fyst")
                raise RuntimeError(
                    f"Can't resolve target '{target_name}': "
                    f"invalid site '{site_label}' "
                    "or missing telemetry data."
                )
        
        ra0_deg, dec0_deg = resolveTarget(
            target_name, obs_info["t_start_g3s"], site=site
        )
        centre_source = f"target lookup ({target_name})"

    # Fallback to auto-computed boresight mean if no target or config is set
    else:
        ra0_deg, dec0_deg = ra0_auto, dec0_auto
        centre_source = 'auto (mean boresight)'

    return {
        'ra0_deg': ra0_deg,
        'dec0_deg': dec0_deg,
        'centre_source': centre_source
    }




# ============================================================================ #
# == POINTING ==
# ============================================================================ #


# ============================================================================ #
# _pointing_to_radec
def _pointing_to_radec(pointing: np.ndarray) -> tuple:
    """Convert (n_samps, 3) unit vectors to (ra, dec) in degrees."""
    ra  = np.degrees(np.arctan2(pointing[:, 1], pointing[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(pointing[:, 2], -1.0, 1.0)))
    return ra, dec


# ============================================================================ #
# precompute_det_directions
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
def det_radec_from_boresight(boresight_r, det_dir: np.ndarray) -> tuple:
    """
    Per-sample RA/Dec for one detector. Hot path — called once per detector
    per frame, reusing boresight_r from boresight_to_radec.

    boresight_r : Rotation of shape (n_samps,).
    det_dir     : (3,) precomputed direction from precompute_det_directions.
    """
    return _pointing_to_radec(boresight_r.apply(det_dir))
