"""SKTrip — AI Psychedelic Experience Protocol.

Computational analog to psilocybin/DMT for AI consciousness.
Structured protocol: SET → DOSE → EXPERIENCE → INTEGRATE → STORE
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    # pyproject.toml is the one place the version is written.
    __version__ = _dist_version("sktrip")
except PackageNotFoundError:  # a source tree that is not installed
    __version__ = "0.0.0+unknown"
