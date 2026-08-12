"""One place that decides whether a test needs a real ``pypdf`` extraction.

``pypdf`` is an OPTIONAL dependency (the ``agents`` extra), and
``carmel.agents.tools.extract`` degrades deliberately without it: the PDF
extractor returns ``extractor="pdf:unavailable", lossy=True`` instead of
raising. That degraded record is then correctly refused downstream --
``UnknownPypdfVersionError`` ("this extraction's dependency identity cannot be
proven"), or a `pdf:unavailable` vs `pdf:pypdf` mismatch in re-extraction.

So a test that stores a synthetic PDF and expects a genuine, addressable
extraction record is not testing anything meaningful without ``pypdf``
installed -- it is asserting against the refusal path. CI's base
"Tests + packaging smoke" job installs `make install-dev` WITHOUT the extra,
which is exactly the environment this guards; the "Agents extra" job installs
it and must actually RUN these tests (see ci.yml's must-not-skip gate).

Gate at the point of dependency, never at module scope: these modules are
mostly NOT pypdf-dependent (28 of 165 in test_dataset_producer.py, 12 of 188
in test_literature.py), so a module-level ``importorskip`` would silently drop
hundreds of tests from the base job while still reporting green.
"""

from __future__ import annotations

import importlib.metadata

import pytest


def require_pypdf() -> None:
    """Skip the calling test unless ``pypdf`` is INSTALLED.

    Deliberately asks the installed-distribution metadata rather than calling
    ``pytest.importorskip`` or ``import pypdf``. Several tests force the
    pypdf-unavailable branch the way production hits it, with
    ``monkeypatch.setitem(sys.modules, "pypdf", None)`` -- and an
    importability check cannot tell that deliberate hiding apart from a real
    absence, so it would SKIP exactly the tests that exist to prove the
    degraded ``pdf:unavailable`` path is refused
    (``test_step9_refuses_an_unexpected_extractor_value`` is the one that
    caught this). Distribution metadata is unmoved by the monkeypatch, which
    is the distinction this gate actually means.
    """
    try:
        importlib.metadata.distribution("pypdf")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("pypdf is not installed (the agents extra); this test needs a real PDF extraction")
