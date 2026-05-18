import json
from typing import Any


def parse_override_value(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return raw


def apply_config_overrides(config_data: dict, overrides: list[str] | None) -> None:
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Config override must be KEY=VALUE, got: {override}")
        key_path, raw_value = override.split("=", 1)
        keys = [part for part in key_path.split(".") if part]
        if not keys:
            raise ValueError(f"Config override key is empty: {override}")

        cursor = config_data
        for key in keys[:-1]:
            child = cursor.get(key)
            if child is None:
                child = {}
                cursor[key] = child
            if not isinstance(child, dict):
                raise ValueError(
                    f"Cannot set {key_path}: {key} is {type(child).__name__}, not an object"
                )
            cursor = child

        cursor[keys[-1]] = parse_override_value(raw_value)
