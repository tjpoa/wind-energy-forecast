# Phase 2 ERA5-Land Grid Readiness

## Scope

This note records the Phase 2 Step 2A.13 grid-readiness policy and bounded July 2023 readiness pilot for ERA5-Land v2 weather ingestion.

It does not perform historical backfill, start Step 2B, regenerate features, refit scalers, retrain models, execute notebooks, or start Phase 3.

## Root Cause Evidence

The ignored Step 2A.12 status at `data/raw/v2/weather/era5_land/metadata/station_id=1200551/period=2023-07-01_2023-07-31/status.json` records a completed CDS retrieval for station `1200551` and period `2023-07-01` through `2023-07-31`, but the requested nearest grid cell returned all-null required variables.

Recorded request evidence:

| Field | Value |
| --- | --- |
| station_id | `1200551` |
| station coordinate | `41.648875`, `-8.804606` |
| requested ERA5-Land area | `[41.6, -8.8, 41.6, -8.8]` |
| period | `2023-07-01` through `2023-07-31` |
| validation status | `invalid` |
| null evidence | 744 null values for each required weather value derived from `2m_temperature`, `10m_u_component_of_wind`, and `10m_v_component_of_wind` |

Interpretation: the v2 point-extraction contract needs an explicit, deterministic policy for coastal or otherwise invalid nearest ERA5-Land grid cells. The station coordinate remains the station evidence; the selected ERA5 grid coordinate is separate operational metadata.

## Evaluated Alternatives

| Alternative | Decision | Rationale |
| --- | --- | --- |
| Keep single-cell nearest grid only | Preserved only as default/radius-0 compatibility | It reproduces Step 2A.12 behavior but can block valid coastal stations. |
| Search an unbounded neighbourhood | Rejected | It expands spatial ambiguity and request count beyond the approved readiness step. |
| Download full regional grids | Rejected for Step 2A.13 | The v2 contract selected point extraction; full-grid extraction is a separate future decision. |
| Silently impute or clean all-null cells | Rejected | Validation must remain separate from cleaning and must not hide source-grid failures. |
| Deterministic nearest-valid 3x3 search | Selected | It is bounded, auditable, preserves station coordinates, and records the selected grid coordinate and distance. |

## Final Policy

Step 2A.13 adds optional grid policy `nearest-valid` with `--grid-search-radius 1`.

Rules:

- Candidate set is bounded to one ERA5-Land grid step around the nearest rounded grid cell, for at most 9 candidates.
- Candidate ordering is deterministic: haversine distance ascending, then absolute latitude delta, absolute longitude delta, grid latitude, and grid longitude.
- One bounded neighbourhood NetCDF is retrieved per station/chunk, then each candidate cell is evaluated from that shared file against the existing hourly and daily validation rules.
- Required variables must not be all-null or partially invalid.
- The first valid candidate is selected.
- The station latitude and longitude remain the requested station coordinates in normalized outputs.
- The selected ERA5 grid latitude, grid longitude, selected-candidate rank, and station-to-grid distance are recorded in status metadata and manifest metadata.
- Default single-cell/radius-0 behavior and paths are preserved for compatibility.
- Radius-1 nearest-valid outputs use policy-specific paths under `grid_policy=nearest_valid_r1` so prior Step 2A.12 evidence is not overwritten.
- Candidate evidence points to the shared raw neighbourhood NetCDF rather than separate per-candidate downloads.

Readiness statuses:

| Selection outcome | Status |
| --- | --- |
| nearest grid candidate validates | `READY` |
| non-nearest candidate validates | `READY_WITH_WARNING` |
| no candidate validates | `BLOCKED` |

## Maximum Search Radius And Distance

The approved search radius is one ERA5-Land grid step. ERA5-Land grid spacing for this ingestion helper is `0.1` degree, so the candidate neighbourhood is a 3x3 grid around the nearest rounded station point.

The exact station-to-grid distance is calculated per selected candidate using haversine distance and recorded as `station_to_grid_distance_km`. A fixed kilometre maximum is not hardcoded because longitude spacing varies with latitude; the operational bound is the one-step 3x3 candidate set.

## Bounded Readiness Pilot

The bounded readiness pilot used:

- Dataset: `reanalysis-era5-land`
- Period: `2023-07-01` through `2023-07-31`
- Variables: `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`
- Station set: all 17 approved exact-match IPMA station coordinates
- Grid policy: `nearest-valid`
- Search radius: `1`
- Output path: `data/raw/v2/weather/era5_land/grid_policy=nearest_valid_r1/`

The diagnostic run for station `1200551` made one bounded 3x3 neighbourhood request with area `[41.7, -8.9, 41.5, -8.7]`. The nearest candidate `41.6, -8.8` remained all-null, while candidate rank `1` at `41.7, -8.8` validated successfully.

The full-station readiness run used `--resume`, reused the verified `1200551` diagnostic partition, and made 16 additional station/month CDS requests. The aggregate output contains 31 daily rows with `point_count=17`, `expected_point_count=17`, and `missing_point_count=0` for every day.

Readiness summary:

| Status | Count |
| --- | ---: |
| `READY` | 15 |
| `READY_WITH_WARNING` | 2 |
| `BLOCKED` | 0 |

| station_id | period | grid_policy | search_radius | readiness_status | selected_grid | station_to_grid_distance_km | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1200545` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `41.2, -8.7` | `4.039` | nearest candidate valid |
| `1200548` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `40.2, -8.5` | `6.232` | nearest candidate valid |
| `1200551` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY_WITH_WARNING` | `41.7, -8.8` | `5.698` | nearest candidate all-null; selected rank `1` |
| `1200554` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `37.0, -8.0` | `3.099` | nearest candidate valid |
| `1200558` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `38.5, -7.9` | `3.982` | nearest candidate valid |
| `1200560` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `40.7, -7.9` | `1.695` | nearest candidate valid |
| `1200562` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `38.0, -7.9` | `3.988` | nearest candidate valid |
| `1200567` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `41.3, -7.7` | `3.206` | nearest candidate valid |
| `1200570` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `39.8, -7.5` | `4.673` | nearest candidate valid |
| `1200571` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `39.3, -7.4` | `1.791` | nearest candidate valid |
| `1200575` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `41.8, -6.7` | `3.577` | nearest candidate valid |
| `1210622` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `41.6, -8.5` | `5.560` | nearest candidate valid |
| `1210683` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `40.5, -7.3` | `4.656` | nearest candidate valid |
| `1210702` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `40.6, -8.7` | `5.200` | nearest candidate valid |
| `1210718` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `39.8, -8.8` | `2.662` | nearest candidate valid |
| `1210734` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY` | `39.2, -8.7` | `3.162` | nearest candidate valid |
| `1210770` | `2023-07-01_2023-07-31` | `nearest-valid` | `1` | `READY_WITH_WARNING` | `38.6, -8.9` | `5.787` | nearest candidate all-null; selected rank `1` |

## Manifest And Status Evidence

For each partition, status JSON includes:

- `grid_selection.grid_policy`
- `grid_selection.grid_search_radius`
- `grid_selection.readiness_status`
- `grid_selection.selected_candidate_rank`
- `grid_selection.selected_grid_coordinate`
- `grid_selection.station_to_grid_distance_km`
- `grid_selection.candidate_evidence`

Manifest metadata includes the same policy contract and partition-level grid-selection evidence.

## Unresolved Risks

- Neighbour selection may introduce a small spatial shift for affected stations. The selected grid coordinate and distance must be reviewed before full historical ingestion.
- The July 2023 readiness pilot proves operational readiness for the sampled calendar month, but it does not prove that every historical month will have identical candidate validity.
- ERA5-Land remains a v2 weather data contract and is not compatible with v1 scalers or trained models without v2 refitting, retraining, and re-baselining.
- CDS service behavior, licensing, accepted terms, and operational availability remain external dependencies for live ingestion.

## Historical Backfill Decision

The approved 17-station ERA5-Land grid policy is operationally ready for a future historical ingestion command because all 17 approved stations validated for the bounded July 2023 readiness pilot and no station remains `BLOCKED`.

Historical backfill was not started by this checkpoint. A future backfill must use the documented `nearest-valid` radius-1 policy, preserve v1 data, write only to approved v2 paths, and keep manifest/status evidence for every station/month partition.

No full historical backfill, Step 2B work, feature regeneration, scaler refitting, model training, notebook execution, or Phase 3 work was started by Step 2A.13.
