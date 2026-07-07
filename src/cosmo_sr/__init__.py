"""cosmo_sr: scarce-HR / many-LR super-resolution for cosmological fields.

The package keeps our own method (fixed degrader ``A``, deterministic generator
``G``, ambient LR-consistency loss, scarce paired-HR loss, evaluation) separate
from the external ``map2map`` / ``SRS-map2map`` repositories, which are used only
as read-only dependencies.
"""

__version__ = "0.1.0"

CANONICAL_CHANNELS = 6
DISP_CHANNELS = slice(0, 3)
VEL_CHANNELS = slice(3, 6)
