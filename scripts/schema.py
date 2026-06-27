"""Compatibility wrapper for the shared schema helpers.

Existing scripts and notebooks may still use ``import schema`` from the
``scripts`` directory. The canonical implementation now lives in
``wind_forecast.schemas``.
"""

from wind_forecast.schemas import *  # noqa: F401,F403
