"""Shared schema and validator for every frozen historical-revision benchmark."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, TypeAdapter

from carmel.benchmarks.n_heptane import Contract as HeptaneContract
from carmel.benchmarks.n_pentane import Contract as PentaneContract

BenchmarkContract = Annotated[HeptaneContract | PentaneContract, Field(discriminator="case")]
BENCHMARK_CONTRACT_ADAPTER: TypeAdapter[BenchmarkContract] = TypeAdapter(BenchmarkContract)


def validate_contract_json(data: bytes | str) -> BenchmarkContract:
    return BENCHMARK_CONTRACT_ADAPTER.validate_json(data)


def benchmark_contract_schema() -> dict[str, object]:
    return BENCHMARK_CONTRACT_ADAPTER.json_schema()
