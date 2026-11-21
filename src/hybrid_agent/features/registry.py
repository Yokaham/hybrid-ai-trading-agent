"""Feature metadata registration and discovery for quantitative signals."""

from typing import Any

from pydantic import BaseModel, Field


class FeatureMetadata(BaseModel, frozen=True):
    """Specification metadata defining a feature's inputs, parameters, warm-up, and outputs."""

    name: str
    description: str
    category: str  # TREND, MOMENTUM, VOLATILITY, VOLUME
    required_inputs: list[str]  # e.g., ["close"] or ["high", "low", "close"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    warmup_period: int = Field(ge=0)
    output_columns: list[str]
    version: str = "1.0.0"


class FeatureRegistry:
    """Central registry tracking all mathematical features and their technical contracts."""

    def __init__(self) -> None:
        self._registry: dict[str, FeatureMetadata] = {}

    def register(self, metadata: FeatureMetadata) -> None:
        """Registers a feature specification."""
        key = metadata.name.lower()
        if key in self._registry:
            raise ValueError(f"Feature '{metadata.name}' is already registered.")
        self._registry[key] = metadata

    def get(self, name: str) -> FeatureMetadata:
        """Retrieves feature metadata by identifier."""
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Feature '{name}' not found in registry.")
        return self._registry[key]

    def list_all(self) -> list[FeatureMetadata]:
        """Returns all registered features sorted by category and name."""
        return sorted(self._registry.values(), key=lambda f: (f.category, f.name))

    def generate_catalog_markdown(self) -> str:
        """Generates markdown documentation for all registered features."""
        lines = [
            "# Feature Catalog & Technical Registry",
            "",
            "| Feature Name | Category | Inputs | Parameters | Warm-up | Outputs | Version |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for f in self.list_all():
            params_str = ", ".join(f"{k}={v}" for k, v in f.parameters.items()) or "None"
            lines.append(
                f"| `{f.name}` | {f.category} | `{', '.join(f.required_inputs)}` | `{params_str}` | {f.warmup_period} | `{', '.join(f.output_columns)}` | {f.version} |"
            )
        return "\n".join(lines)


# Global default feature registry instance
default_feature_registry = FeatureRegistry()
