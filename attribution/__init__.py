"""Portfolio return attribution for Indian equity portfolios.

Decision view (macro exposure timing, allocation, selection, interaction, then
fees and trading costs) and factor view (six factors plus specific return),
both reconciling to the same active return. Start with
``attribution.engine.run_attribution``; see README.md.
"""

from attribution.result import SCHEMA_VERSION, AttributionResult
from attribution.schemas import FACTORS, SECTORS, Policy, SchemaValidationError

__version__ = "0.2.0"

__all__ = ["__version__", "SCHEMA_VERSION", "AttributionResult", "Policy", "SchemaValidationError", "SECTORS", "FACTORS"]
