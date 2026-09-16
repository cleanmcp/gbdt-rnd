from pathlib import Path

from signal_engine.signal_registry import SignalRegistry


def test_new_signal_is_added_by_validated_declaration(tmp_path: Path) -> None:
    declaration = tmp_path / "machine-maintenance.yaml"
    declaration.write_text(
        """
schema_version: 1
signal_type: machine_maintenance
version: 1
description: A machine maintenance lifecycle.
shape: state_machine
source_ids: [fixture]
durable: true
identity_fields: [machine_id]
state_field: status
terminal_states: [resolved]
material_kinds: [opened, escalated, resolved]
numeric_features:
  - field: downtime_hours
    unit: hours
    reducers: [latest, max, sum]
default_half_life_days: 30
applicability_naics_prefixes: ["31", "32", "33"]
""".strip(),
        encoding="utf-8",
    )
    registry = SignalRegistry.from_directory(tmp_path)
    definition = registry.get("machine_maintenance")
    assert definition.shape == "state_machine"
    assert definition.contract_hash == registry.manifest()["machine_maintenance@1"]
