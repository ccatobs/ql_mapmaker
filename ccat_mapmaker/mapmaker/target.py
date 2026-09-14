# ============================================================================ #
# target.py
#
# James Burgoyne, jburgoyne@phas.ubc.ca
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT August 2026
#
# Get the a map centre (RA/Dec) from a target name, based off mmi_map_lib.py
# ============================================================================ #

from typing import Optional

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, get_body
import astropy.units as u

# FYST site on Cerro Chajnantor for simulations and when there is real data
FYST_SITE = EarthLocation(lat=-22.98592 * u.deg, lon=-67.74028 * u.deg, height=5610 * u.m)

_SOLAR_SYSTEM_BODIES = {
    "sun", "moon", "mercury", "venus", "mars",
    "jupiter", "saturn", "uranus", "neptune",
}


# ============================================================================ #
# is_solar_system_body
# ============================================================================ #
def is_solar_system_body(name: str) -> bool:
    return name.strip().lower() in _SOLAR_SYSTEM_BODIES


# ============================================================================ #
# resolve_target
# ============================================================================ #
def resolve_target(name: str, obs_time_unix: Optional[float] = None,
                   site: Optional[EarthLocation] = None) -> tuple[float, float]:
    """
    Resolve a target name to (ra_deg, dec_deg).
    """
    if is_solar_system_body(name):
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
