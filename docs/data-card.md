# AmbientSenseBench data card

## Purpose

AmbientSenseBench supports reproducible evaluation of ambient IoT pipelines when using
real home-monitoring data is inappropriate or unavailable. It emits synthetic event traces,
then derives daily features with the same extraction pipeline used by downstream models.

## Event schema

Each raw-event CSV has three columns: `timestamp`, `sensor_id`, and `state`.

The included sensor identifiers represent PIR motion sensors, door contacts, and appliance
state changes. They are a compact research schema, not a capture from a specific vendor or
an asserted Zigbee/MQTT wire format. Adapters can map this schema to deployment-specific
messages.

## Scenario labels

The `baseline` and `distress` labels identify simulated routine states. They must not be
interpreted as clinical labels, diagnoses, or a basis for individual decision-making.

## Limitations

- Parameters encode explicit modelling assumptions rather than real-participant estimates.
- Synthetic events do not establish external validity for a home deployment.
- The reference scenarios are starting points, not a representative benchmark population.
- Any extension to a new application requires new features, profiles, and validation rules.

