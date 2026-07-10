"""JSON Schema validation with a dependency-free fallback for core tool schemas."""

from __future__ import annotations

from typing import Any, Dict


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(expected, True)


def _basic_validate(value: Any, schema: Dict[str, Any], path: str = "$" ) -> None:
    if "anyOf" in schema:
        errors = []
        for choice in schema["anyOf"]:
            try:
                _basic_validate(value, choice, path)
                return
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(f"{path} does not match any allowed schema")
    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected] if expected else []
    if expected_types and not any(_matches_type(value, item) for item in expected_types):
        raise ValueError(f"{path} must have type {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                raise ValueError(f"{path}.{required} is required")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise ValueError(f"{path} contains unsupported properties: {sorted(extras)}")
        for key, item in value.items():
            if key in properties:
                _basic_validate(item, properties[key], f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _basic_validate(item, schema["items"], f"{path}[{index}]")


def validate_json(value: Any, schema: Dict[str, Any]) -> None:
    """Validate with jsonschema when installed, otherwise cover core OpenTools schemas."""
    try:
        import jsonschema
    except ImportError:
        _basic_validate(value, schema)
        return
    jsonschema.validate(instance=value, schema=schema)


__all__ = ["validate_json"]
