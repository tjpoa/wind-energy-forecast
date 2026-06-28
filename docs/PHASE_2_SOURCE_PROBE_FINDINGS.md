# Phase 2 Source Probe Findings

## 1. Purpose

This document records the Phase 2 Step 2A.7 REN and IPMA source-probe findings before any v2 dataset is created. It preserves the evidence needed to decide whether official production and station-metadata sources can support a future v2 data contract.

This is a findings document only. It does not select a final v2 source, modify v1 data, or start the ERA5-Land pilot.

## 2. Scope And Safety

Only limited official-source probes were run:

- REN: three explicit single-date requests.
- IPMA: current station metadata resources only.
- No bulk history was downloaded.
- No ERA5-Land, Copernicus CDS, notebook, pipeline, training, or inference execution occurred.
- Probe outputs were written only under ignored `data/pilot/`.
- Existing v1 raw data, processed data, models, scalers, notebooks, and baselines were not changed.

The probe tooling was committed in `03760fb1 feat: add REN and IPMA source probes`.

## 3. REN Endpoint Evidence

Confirmed API observations:

| Item | Evidence |
| --- | --- |
| Endpoint identifier | `REN ElectricityProductionBreakdownDaily` |
| Request pattern | One date per request with `culture=pt-PT` and `date=YYYY-MM-DD` |
| Tested dates | `2010-01-01`, `2025-01-25`, `2026-06-27` |
| Daily record count | 96 records on each tested complete day |
| Returned cadence | 15 minutes, inferred from `00:00` to `23:45` |
| Wind series | `Eólica` |
| Unit exposed by response | `MW` |
| Top-level response keys | `xAxis`, `yAxis`, `series` |

Confirmed local comparison results:

| Date | Aligned Timestamps | Exact Matches | MAE | Max Absolute Difference | Correlation | Median REN/Local Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2010-01-01` | 96 | 96 | `0.0` | `0.0` | `1.0` | `1.0` |
| `2025-01-25` | 96 | 0 | `15.21666666666667 MW` | `18.200000000000728 MW` | `0.9999960649429729` | `1.005018136527911` |

The `2026-06-27` probe returned 96 records at 15-minute cadence with the `Eólica` series and `MW` units. No local v1 comparison exists for that date because it is beyond the current v1 production-file coverage.

Interpretation:

- The REN endpoint appears compatible with the v1 production target shape: 15-minute wind production values in MW.
- The exact `2010-01-01` overlap is strong evidence of target compatibility.
- The `2025-01-25` overlap shows that later overlapping values can differ and must not be silently appended to the frozen v1 snapshot.

## 4. Production-Source Conclusion

The exact `2010-01-01` correspondence provides strong evidence that the v1 wind-production target is compatible with the official REN source and contract.

This does not prove that every historical v1 value has been verified. It also does not prove the original CSV download mechanism. The current repository evidence supports REN as the leading official v2 production-source candidate, but the full historical contract still requires more verification before v2 selection.

## 5. Historical-Revision Finding

The `2025-01-25` overlap aligned on all 96 timestamps but differed slightly. The correlation remained very high, but the values were not exact.

Unresolved hypotheses:

- Later REN revisions changed values after the local CSV snapshot.
- The local CSV contained provisional values while the current API returns definitive values, or the reverse.
- The local CSV and API were retrieved at different source snapshots.
- Another upstream update process changed overlapping values.
- Less likely: parsing, encoding, timezone, or transformation differences in the probe.

No single explanation is selected because the available evidence does not identify the cause.

## 6. V2 Production Implication

A future v2 production dataset should provisionally be retrieved consistently from the official REN source if further verification confirms the contract.

The v2 production process should record:

- Endpoint identifier and request parameters.
- Retrieval timestamp.
- Units and temporal granularity.
- Raw response checksums.
- Coverage start and end.
- Whether values are provisional or definitive where discoverable.
- Overlap-comparison results against frozen v1 snapshots.

Direct append to v1 is rejected for now. Overlapping values may change, so future work must choose explicitly between:

- full historical reconstruction from a consistent source snapshot;
- controlled append with revision tracking and clear boundary rules.

The v1 dataset remains immutable.

## 7. Recent-Data Availability

The `2026-06-27` REN probe returned successfully. This proves that official REN data exists beyond the current v1 production end date of `2025-04-28`.

This single successful recent date does not prove complete coverage for every intervening date.

## 8. IPMA Mapping Results

Confirmed mapping results from current IPMA metadata:

| Item | Result |
| --- | --- |
| Total v1 station identifiers | 18 |
| Exact current metadata matches | 17 |
| Ambiguous matches | 0 |
| Unmatched identifiers | 1 |
| Unmatched ID | `1200579` |
| Coordinates available for pilot | Yes, for the 17 matched identifiers |

The matched metadata provides current station names and coordinates suitable for a limited ERA5-Land pilot. Raw metadata dumps are intentionally not copied into this document.

## 9. Station-Mapping Limitations

Present-day IPMA identifier correspondence does not prove IPMA was the original historical-weather source.

Limitations:

- Station coordinates or metadata may have changed historically.
- The unmatched `1200579` identifier must not be guessed.
- The 17 confirmed coordinates can support a v2 pilot, but the missing station must be documented.
- Alternative spatial strategies remain possible, including a documented regular grid or representative coordinate set.

## 10. ERA5-Land Pilot Readiness

The source probes establish enough evidence to begin a limited ERA5-Land technical pilot using:

- one confidently mapped station;
- a one-week period in 2023;
- `2m_temperature`;
- `10m_u_component_of_wind`;
- `10m_v_component_of_wind`.

The ERA5-Land pilot must happen before any full download, v2 source contract selection, feature regeneration, scaler refitting, or model retraining.

## 11. Decision Register

| Decision | Status | Evidence |
| --- | --- | --- |
| Preserve v1 unchanged | Approved | Existing baseline and possible historical revisions |
| Use REN as leading v2 production source | Provisional | Exact `2010-01-01` overlap and recent availability |
| Append directly to v1 | Rejected for now | Overlap differences on `2025-01-25` |
| Use 17 mapped coordinates for pilot | Approved | Exact current IPMA metadata matches |
| Guess mapping for `1200579` | Rejected | No exact IPMA metadata match |
| Select ERA5-Land as final v2 weather source | Not yet decided | Pilot still required |

## 12. Remaining Questions

- Why do the `2025-01-25` overlapping REN values differ?
- Does REN expose provisional/final status for the endpoint or data points?
- What is the earliest continuously available comparable REN date?
- Are all dates after `2025-04-28` available with the same contract?
- What was the historical status of the IPMA station metadata during v1 coverage?
- What is the identity or replacement history of `1200579`?
- Are ERA5-Land distributions sufficiently compatible with the intended v2 model?
- Which spatial strategy and daily aggregation formulas should be adopted?

## 13. Next Step

The next approved activity is `Phase 2 — Step 2A.8: ERA5-Land one-point technical pilot`.

Step 2B remains paused. Phase 3 was not started.
