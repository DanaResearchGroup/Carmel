"""Shared fixture for building a valid figure-digitization citation in tests.

Every envelope fixture carrying a ``SourceForm.DIGITIZED`` series now has to
satisfy V9/FD1-FD3 (see :mod:`carmel.schemas.datasets`), which means minting a
real :class:`~carmel.schemas.datasets.EmbeddedFigureDigitization` whose canonical
bytes hash to the address the series cites. Doing that once, here, keeps each
test module from growing its own near-copy -- the same reason
:mod:`tests.table_inventory_fixtures` exists for the table lane, and the same
"match the table lane rather than reinvent it" principle the citation surface
itself was built on.

Two near-copies of this helper is not a hypothetical drift risk: the record shape
is bound to the validators by *content hash*, so a copy that falls a field behind
does not fail loudly at the seam -- it mints a different address and fails
somewhere else entirely.
"""

from __future__ import annotations

import hashlib

from carmel.schemas.datasets import EmbeddedFigureDigitization, Series, SourceGraph
from carmel.services.figure_digitization_record import (
    FigureCoverage,
    FigureDigitization,
    MarkerCensus,
    PlotRegion,
    digitization_record_bytes,
    digitization_record_payload,
)


def cite_digitization(
    series: Series, graph: SourceGraph, crop_node_id: str
) -> tuple[Series, EmbeddedFigureDigitization]:
    """Return ``(series_with_citation, embedded)`` for a DIGITIZED ``series``
    recovered from ``crop_node_id``, a real resolvable citation the V9/FD
    validators accept.

    ``raw_sha256`` is taken from the crop's ROOT ancestor, not its immediate
    parent, because that is exactly what V9 joins on -- a fixture that used the
    parent would pass construction and fail validation for a reason that looks
    unrelated.
    """
    crop = graph.node(crop_node_id)
    ancestors = graph.ancestors(crop_node_id)
    root = ancestors[-1] if ancestors else crop
    recovered = len(series.points)
    record = FigureDigitization(
        series_id=series.series_id,
        raw_sha256=root.sha256,
        figure_crop_node_id=crop_node_id,
        figure_crop_sha256=crop.sha256,
        plot_region=PlotRegion(page=1, x_start=72.0, x_end=520.0, y_bottom=100.0, y_top=640.0),
        coverage=FigureCoverage.COMPLETE,
        census=MarkerCensus(detected=recovered),
        recovered=recovered,
        omissions=(),
    )
    canonical = digitization_record_bytes(digitization_record_payload(record))
    sha = hashlib.sha256(canonical).hexdigest()
    embedded = EmbeddedFigureDigitization(
        digitization_sha256=sha, raw_sha256=root.sha256, canonical_json=canonical.decode("utf-8")
    )
    return series.model_copy(update={"digitization_sha256": sha}), embedded
