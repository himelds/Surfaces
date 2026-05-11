# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Deprecated compatibility namespace for BBOB and CEC test functions.

Use :mod:`surfaces.test_functions.bbob` and :mod:`surfaces.test_functions.cec`
instead. The :mod:`surfaces.benchmark` package contains the benchmark runner.
"""

from __future__ import annotations

import sys
import warnings
from importlib import import_module

from ..bbob import (
    BBOB_FUNCTIONS,
    # Low/Moderate Conditioning (f6-f9)
    AttractiveSector,
    BBOBFunction,
    BentCigar,
    BuecheRastrigin,
    DifferentPowers,
    Discus,
    # High Conditioning & Unimodal (f10-f14)
    EllipsoidalRotated,
    EllipsoidalSeparable,
    Gallagher21,
    Gallagher101,
    GriewankRosenbrock,
    Katsuura,
    LinearSlope,
    LunacekBiRastrigin,
    # Multimodal with Adequate Global Structure (f15-f19)
    RastriginRotated,
    RastriginSeparable,
    RosenbrockOriginal,
    RosenbrockRotated,
    SchaffersF7,
    SchaffersF7Ill,
    # Multimodal with Weak Global Structure (f20-f24)
    Schwefel,
    SharpRidge,
    # Separable (f1-f5)
    Sphere,
    StepEllipsoidal,
    Weierstrass,
    bbob_functions,
)
from ..cec import CECFunction

warnings.warn(
    "surfaces.test_functions.benchmark is deprecated; use "
    "surfaces.test_functions.bbob or surfaces.test_functions.cec instead.",
    DeprecationWarning,
    stacklevel=2,
)

_MODULE_ALIASES = {
    "_batch_transforms": "surfaces.test_functions._batch_transforms",
    "bbob": "surfaces.test_functions.bbob",
    "bbob._base_bbob": "surfaces.test_functions.bbob._base_bbob",
    "bbob.high_conditioning": "surfaces.test_functions.bbob.high_conditioning",
    "bbob.low_conditioning": "surfaces.test_functions.bbob.low_conditioning",
    "bbob.multimodal_adequate": "surfaces.test_functions.bbob.multimodal_adequate",
    "bbob.multimodal_weak": "surfaces.test_functions.bbob.multimodal_weak",
    "bbob.separable": "surfaces.test_functions.bbob.separable",
    "cec": "surfaces.test_functions.cec",
    "cec._base_cec": "surfaces.test_functions.cec._base_cec",
    "cec._data_utils": "surfaces.test_functions.cec._data_utils",
    "cec.cec2013": "surfaces.test_functions.cec.cec2013",
    "cec.cec2013._base_cec2013": "surfaces.test_functions.cec.cec2013._base_cec2013",
    "cec.cec2013.functions": "surfaces.test_functions.cec.cec2013.functions",
    "cec.cec2014": "surfaces.test_functions.cec.cec2014",
    "cec.cec2014._base_cec2014": "surfaces.test_functions.cec.cec2014._base_cec2014",
    "cec.cec2014.composition": "surfaces.test_functions.cec.cec2014.composition",
    "cec.cec2014.hybrid": "surfaces.test_functions.cec.cec2014.hybrid",
    "cec.cec2014.multimodal": "surfaces.test_functions.cec.cec2014.multimodal",
    "cec.cec2014.unimodal": "surfaces.test_functions.cec.cec2014.unimodal",
    "cec.cec2017": "surfaces.test_functions.cec.cec2017",
    "cec.cec2017._base_cec2017": "surfaces.test_functions.cec.cec2017._base_cec2017",
    "cec.cec2017.simple": "surfaces.test_functions.cec.cec2017.simple",
}

for _old_suffix, _new_name in _MODULE_ALIASES.items():
    sys.modules[f"{__name__}.{_old_suffix}"] = import_module(_new_name)

cec_functions = []  # CEC functions are loaded dynamically

__all__ = [
    # Base classes
    "BBOBFunction",
    # BBOB - Separable
    "Sphere",
    "EllipsoidalSeparable",
    "RastriginSeparable",
    "BuecheRastrigin",
    "LinearSlope",
    # BBOB - Low/Moderate Conditioning
    "AttractiveSector",
    "StepEllipsoidal",
    "RosenbrockOriginal",
    "RosenbrockRotated",
    # BBOB - High Conditioning & Unimodal
    "EllipsoidalRotated",
    "Discus",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    # BBOB - Multimodal Adequate
    "RastriginRotated",
    "Weierstrass",
    "SchaffersF7",
    "SchaffersF7Ill",
    "GriewankRosenbrock",
    # BBOB - Multimodal Weak
    "Schwefel",
    "Gallagher101",
    "Gallagher21",
    "Katsuura",
    "LunacekBiRastrigin",
    # Function lists
    "bbob_functions",
    "BBOB_FUNCTIONS",
    "cec_functions",
]

benchmark_functions = bbob_functions + cec_functions
