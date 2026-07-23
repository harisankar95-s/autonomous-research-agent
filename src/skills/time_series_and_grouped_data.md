---
name: time_series_and_grouped_data
description: Per-entity vs pooled baselines, timestamp gap and sampling-frequency checks, timezone alignment, unit/scale comparability across entities. Load this when the dataset has a timestamp column and/or a unit/entity identifier (multiple devices, stores, patients, sensors, or similar) with more than one distinct value.
---

These standards apply whenever a dataset is time-indexed, contains multiple
distinct entities of the same kind, or both. Both conditions are common
together (e.g. a fleet of devices each reporting readings over time), and
each requires its own checks.

PER-ENTITY VS POOLED BASELINES
When a dataset contains multiple instances of the same kind of entity (units,
stores, patients, sensors), "normal" is not necessarily one number for the
whole dataset. Different entities can have legitimately different normal
operating ranges - due to age, model, location, configuration, or scale.
Before flagging a value as anomalous against a pooled, dataset-wide baseline,
check whether the entity that produced it has its own distinct baseline, and
compare against that first. A value that is anomalous relative to the fleet
can be entirely normal for that specific entity, and vice versa - a value
within the fleet-wide range can still be a real anomaly for the specific
entity that produced it.

TIMESTAMP GAP AND FREQUENCY CONSISTENCY
Before treating a time-indexed column as a clean, regularly sampled series,
check the actual distribution of gaps between consecutive timestamps (per
entity, not pooled - see above). Verify the nominal sampling interval you
expect actually holds, identify missing periods or gaps, and check whether
gaps cluster in time or by entity (which usually signals an outage or
collection issue, not an absence of the phenomenon you're studying). Any
rate-based or trend claim computed over an irregularly sampled series without
accounting for these gaps is unreliable.

TIMEZONE AND CLOCK ALIGNMENT
When comparing timestamps across multiple entities or sources, verify they
are actually on the same clock and timezone before treating them as
comparable or before joining/aligning records across entities by time. A
silent timezone or clock-offset mismatch will misrepresent simultaneity and
can manufacture or hide an apparent temporal relationship.

UNIT AND SCALE COMPARABILITY
Before pooling or directly comparing a numeric column across entities,
confirm the values are actually on the same scale and unit for every entity
- different entities occasionally report in different units, precisions, or
calibrations even within a supposedly homogeneous fleet or dataset. Check
this explicitly (e.g. compare each entity's range and typical magnitude)
before combining entities into one baseline or one comparison.
