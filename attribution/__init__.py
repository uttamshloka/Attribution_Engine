"""Portfolio return attribution for Indian equity portfolios.

This release contains the contracts only: input schemas and loaders, a
synthetic sample generator, and the output result schema. See README.md.
"""

from attribution.result import SCHEMA_VERSION, AttributionResult
from attribution.schemas import FACTORS, SECTORS, Policy, SchemaValidationError

__version__ = "0.1.0"

__all__ = ["__version__", "SCHEMA_VERSION", "AttributionResult", "Policy", "SchemaValidationError", "SECTORS", "FACTORS"]
