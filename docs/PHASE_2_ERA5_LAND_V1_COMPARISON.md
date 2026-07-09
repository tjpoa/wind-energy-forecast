# Phase 2 ERA5-Land And V1 Weather Comparison

## Purpose and scope

This document records Phase 2 Checkpoint 3: a read-only comparison between the limited ERA5-Land multi-point pilot and the current v1 weather matrices.

The scope is diagnostic only:

- Compare the existing ERA5-Land pilot outputs with overlapping v1 station-day weather rows.
- Assess schema compatibility, technical data quality, distribution shift, and model/scaler implications.
- Preserve the current v1 raw data, processed data, models, scalers, scripts, notebooks, and pilot artifacts unchanged.
- Do not select a final v2 weather source, regenerate outputs, execute notebooks, run training, or run live CDS/network requests.

This report uses the existing ignored pilot outputs under `data/pilot/era5_land/`, especially `era5_land_multi_point_2023_winter_summer_*`.

## Evidence inputs and checksums

Read-only evidence files:

| File | Role | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `docs/ML_ENGINEERING_ROADMAP.md` | Phase 2 objective and constraints | 17,662 | `32bd36617ddf78ca2b921b15743dde2a4d60069dd811e79d244578d06eca7f83` |
| `docs/PHASE_2_DATA_REFRESH_ASSESSMENT.md` | Data-refresh assessment and ERA5-Land pilot requirements | 17,767 | `d9db760e4875e7f8ee36ca9fdb198719ab26910c04234840838d3d35c61f8f81` |
| `docs/PHASE_2_SOURCE_PROBE_FINDINGS.md` | REN/IPMA source-probe findings and pilot readiness | 7,586 | `7323899a9e27fc9285a7c456db115b8dce0f28c467dc00f3199fc59cc7be58e2` |
| `scripts/pilot_era5_land_one_point.py` | ERA5-Land point extraction formulas and output units | 26,097 | `f51f4e59b02f2c66e2e8b27b0e5c20bf162e9fc947001c0b364ab1bdcf1a3f0a` |
| `scripts/pilot_era5_land_multi_point.py` | Multi-point pilot, v1 comparison, validation, and aggregation logic | 37,102 | `0a44e17e0b35386e7f37a86b31b3f30eb3161964e17ff0266ab7be04d2785f28` |
| `data/raw/IntensidadeMediaVento10m.csv` | v1 wind-speed matrix | 352,665 | `e09e94f07618ddb4d52d0b53f46a69e1d6b64814a142e69ee57161f4199e842c` |
| `data/raw/DirecaoMediaVento10m.csv` | v1 wind-direction matrix | 469,994 | `289aa375b4c6a371ab4ca649b999ee4ac1cb16e29a5efea01ede86bca8ba08bd` |
| `data/raw/TemperaturaMedia.csv` | v1 temperature matrix | 400,139 | `ce2829f328e46a99f942343ff0a5a27f18319b5a1dc5728c5c110692596fcef4` |

Pilot output evidence:

| File | Role | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_hourly.csv` | ERA5-Land hourly pilot rows | 146,328 | `c0e6332c38ade4a0fd19a0fbded8d44efdccec0bcc346058fc6ccfc5489deedf` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_daily_points.csv` | ERA5-Land station-day rows | 10,919 | `9d0c0cc54848f49b9a4302c8f7efbc6b3756ea8f5e405d8ef2c8b42763298741` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_daily_aggregate.csv` | ERA5-Land three-station daily aggregate | 2,213 | `1cfacc2c78b58d9e9aa1a86490f2075e9f65d8a0a46b730feeee1587a430501b` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_season_summary.csv` | Pilot seasonal summary | 1,817 | `c883bc5e2d6d8bc1fb51dd488ff9d69c682b329ac96f4a05b1a14c27e5539f53` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_v1_comparison.csv` | Paired ERA5-Land/v1 station-day comparison | 15,492 | `dd046e7c817e7b47c8d91df42aeb81e9631db43b65326b8209ea03d108b4e39d` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_metadata.json` | Pilot metadata | 21,718 | `78db650dd3673aee4b23ac67aa6540776d6d815ed5ef9d317082f114d5a550ef` |
| `data/pilot/era5_land/era5_land_multi_point_2023_winter_summer_validation.json` | Pilot validation report | 6,715 | `d04738e78257b7c14fe79e92e0bdb4756c09c0009bfd5d2a3fbd0cae64e746cb` |

Raw NetCDF pilot checksums are recorded in the metadata and validation JSON. The six raw keys are:

| Raw key | SHA-256 |
| --- | --- |
| `winter_1210622` | `c3a5e087ed2d412242e7e529b60ea7b018b55523e04ac9222c2067711a9293b2` |
| `winter_1210683` | `350deefb2289d1de4a9c8da8767a2cebfc7f2d02e6db3e51635e345740bee40c` |
| `winter_1200562` | `d4c8fdfa6049f44a432ad59c7f034fdaa4580ed6dc9febadb7d6640801aae90c` |
| `summer_1210622` | `b688212d50a863b0f257d2ad7a5e53d490c0066103cfda948c29d3b61b2fafff` |
| `summer_1210683` | `8c2bd46e11de4a4e22f20d55f3fed011d05eabf8024320a60ecef88dd9483e5a` |
| `summer_1200562` | `9321e854829dfa62edc8666d6ec32fbedcaf6c7354e119be1a579c5706b9ba39` |

## Pilot coverage and schema compatibility

The v1 weather files are daily station matrices with `ANO`, `MES`, `DIA`, and 18 station columns. Read-only checks found 4,017 rows in each matrix, covering `2013-01-01` through `2023-12-31`.

The pilot comparison covers:

| Item | Value |
| --- | ---: |
| ERA5-Land hourly rows | 1,008 |
| ERA5-Land station-day rows | 42 |
| ERA5-Land aggregate daily rows | 14 |
| Paired v1 comparison rows | 42 |
| Seasons | `winter`, `summer` |
| Date windows | `2023-01-01` to `2023-01-07`; `2023-07-01` to `2023-07-07` |
| Stations | `1200562` Beja; `1210622` Braga Merelim; `1210683` Guarda |
| v1 available rows | 42 of 42 |

The pilot validation JSON reports `passed: true`, `issues: []`, and the expected counts above. The multi-point metadata records `service_status: not_contacted_reuse_raw` for the six raw files, meaning the comparison artifacts were rebuilt from existing raw pilot files without contacting CDS during that validation run.

Missingness in the 42 paired rows:

| Column group | Missing values |
| --- | ---: |
| v1 temperature, wind speed, wind direction | 0 |
| ERA5-Land temperature and wind speed | 0 |
| ERA5-Land vector wind direction | 1 |
| Circular wind-direction differences | 1 |

The single missing ERA5-Land vector direction is `Beja` on `2023-01-06`. The daily scalar mean wind speed is `0.962 m/s`, but the vector-mean speed is `0.414 m/s`, below the pilot calm threshold of `0.5 m/s`; the direction is therefore intentionally null rather than imputed.

Schema compatibility is sufficient for a diagnostic station-day comparison: station IDs and dates align, v1 speed is in `m/s`, v1 temperature is in `deg C`, and both sources express wind direction as degrees from. This is structural compatibility for comparison only, not evidence of source equivalence or model compatibility.

## Calculation methodology

ERA5-Land pilot rows use:

- `2m_temperature`, converted from Kelvin to `deg C`.
- `10m_u_component_of_wind` and `10m_v_component_of_wind`.
- Wind speed as `sqrt(u10^2 + v10^2)`.
- Meteorological wind direction from as `(180 + degrees(atan2(u10, v10))) % 360`.
- A calm-or-near-calm threshold of `0.5 m/s`; vector directions below that threshold are null.

Station-day ERA5-Land values come from hourly UTC aggregation. Daily vector direction uses daily mean `u10` and `v10`.

The v1 comparison reads the three raw v1 matrices in read-only mode, creates `date_utc` from `ANO`, `MES`, and `DIA`, melts the three pilot station columns, and joins on `date_utc` and `station_id`.

For temperature and wind speed:

- Difference is `ERA5-Land - v1`.
- Signed mean difference is the mean of that difference.
- Mean absolute difference is the mean absolute value of that difference.
- Pearson and Spearman correlations are calculated on paired non-null rows.

For wind direction, raw-degree correlation is not used as primary evidence. The primary metric is signed circular difference:

```text
((era5_deg - v1_deg + 180) % 360) - 180
```

Absolute circular difference is the absolute value of that signed circular difference.

For aggregate results, ERA5-Land aggregate rows come from `era5_land_multi_point_2023_winter_summer_daily_aggregate.csv`. V1 aggregate rows are computed from the paired station-day comparison:

- Temperature is the daily mean across the three v1 stations.
- Wind speed is the daily mean across the three v1 stations.
- V1 aggregate direction uses speed-weighted vector components matching the project convention:
  - `u = -speed * sin(direction_rad)`
  - `v = -speed * cos(direction_rad)`
  - mean `u` and mean `v` are converted back with `(180 + degrees(atan2(mean_u, mean_v))) % 360`.

No generated outputs were rewritten for this report.

## Station-level comparison

Overall paired station-day temperature and wind-speed metrics:

| Variable | n | ERA5 missing | v1 missing | ERA5 mean | ERA5 std | ERA5 min | ERA5 q25 | ERA5 q50 | ERA5 q75 | ERA5 max | v1 mean | v1 std | v1 min | v1 q25 | v1 q50 | v1 q75 | v1 max | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temperature, deg C | 42 | 0 | 0 | 16.052 | 8.071 | 3.880 | 8.892 | 17.199 | 22.818 | 29.766 | 15.745 | 7.732 | 4.100 | 8.550 | 16.400 | 22.175 | 27.600 | 0.981 | 0.966 | 0.307 | 1.350 |
| Wind speed, m/s | 42 | 0 | 0 | 2.395 | 0.905 | 0.939 | 1.717 | 2.236 | 2.883 | 4.460 | 2.917 | 1.367 | 0.600 | 1.925 | 2.650 | 3.975 | 5.700 | 0.607 | 0.564 | -0.522 | 1.055 |

Temperature by station:

| Station | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1200562` Beja | 14 | 19.196 | 18.221 | 0.997 | 0.974 | 0.975 | 1.386 |
| `1210622` Braga Merelim | 14 | 14.962 | 16.029 | 0.996 | 0.947 | -1.066 | 1.191 |
| `1210683` Guarda | 14 | 13.998 | 12.986 | 0.988 | 0.950 | 1.013 | 1.473 |

Wind speed by station:

| Station | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1200562` Beja | 14 | 2.530 | 3.443 | 0.957 | 0.982 | -0.913 | 0.913 |
| `1210622` Braga Merelim | 14 | 2.556 | 1.757 | 0.898 | 0.771 | 0.799 | 0.799 |
| `1210683` Guarda | 14 | 2.098 | 3.550 | 0.872 | 0.869 | -1.452 | 1.452 |

Temperature by season:

| Season | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Summer | 21 | 23.390 | 22.786 | 0.840 | 0.816 | 0.604 | 1.783 |
| Winter | 21 | 8.714 | 8.705 | 0.924 | 0.913 | 0.009 | 0.917 |

Wind speed by season:

| Season | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Summer | 21 | 2.524 | 3.171 | 0.018 | 0.090 | -0.648 | 1.228 |
| Winter | 21 | 2.266 | 2.662 | 0.787 | 0.691 | -0.396 | 0.882 |

The station-level results show high temperature agreement and mixed wind-speed agreement. Wind speed has good within-station correlations, but the sign and size of bias differ by station. Summer wind-speed correlation across pooled station-days is near zero because station-specific offsets and a short seven-day window dominate the pooled sample.

## Aggregate comparison

Overall aggregate daily temperature and wind-speed metrics:

| Variable | n | ERA5 mean | ERA5 std | ERA5 min | ERA5 q25 | ERA5 q50 | ERA5 q75 | ERA5 max | v1 mean | v1 std | v1 min | v1 q25 | v1 q50 | v1 q75 | v1 max | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temperature, deg C | 14 | 16.052 | 7.861 | 6.633 | 7.937 | 17.026 | 22.381 | 26.094 | 15.745 | 7.562 | 6.533 | 8.117 | 15.933 | 21.500 | 26.100 | 0.997 | 0.969 | 0.307 | 0.617 |
| Wind speed, m/s | 14 | 2.395 | 0.825 | 1.274 | 1.732 | 2.372 | 2.654 | 4.262 | 2.917 | 1.039 | 1.533 | 2.042 | 2.983 | 3.492 | 4.733 | 0.952 | 0.955 | -0.522 | 0.538 |

Aggregate temperature by season:

| Season | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Summer | 7 | 23.390 | 22.786 | 0.999 | 1.000 | 0.604 | 0.606 |
| Winter | 7 | 8.714 | 8.705 | 0.943 | 0.750 | 0.009 | 0.629 |

Aggregate wind speed by season:

| Season | n | ERA5 mean | v1 mean | Pearson | Spearman | Signed mean diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Summer | 7 | 2.524 | 3.171 | 0.636 | 0.857 | -0.648 | 0.680 |
| Winter | 7 | 2.266 | 2.662 | 0.983 | 0.775 | -0.396 | 0.396 |

Aggregation dampens station-level noise. The aggregate wind-speed correlation is high, but ERA5-Land remains lower than v1 by `0.522 m/s` on average across the 14 aggregate days.

## Seasonal profiles

Seasonal station-day profiles:

| Season | Temp ERA5 mean | Temp v1 mean | Temp signed diff | Temp MAD | Speed ERA5 mean | Speed v1 mean | Speed signed diff | Speed MAD | Direction signed circular diff | Direction mean abs circular diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Summer | 23.390 | 22.786 | 0.604 | 1.783 | 2.524 | 3.171 | -0.648 | 1.228 | 18.711 | 21.868 |
| Winter | 8.714 | 8.705 | 0.009 | 0.917 | 2.266 | 2.662 | -0.396 | 0.882 | -19.190 | 26.066 |

Temperature preserves the broad winter/summer profile. Wind speed is lower in ERA5-Land in both sampled seasons. Direction differences are seasonally asymmetric: ERA5-Land is clockwise relative to v1 on average in summer and counter-clockwise in winter, but the sample is too small to treat this as a stable seasonal rule.

## Wind-direction circular-difference analysis

Station-day circular differences:

| Scope | n | Signed mean | Signed std | Signed min | Signed q25 | Signed q50 | Signed q75 | Signed max | Abs mean | Abs q50 | Abs q75 | Abs q95 | Abs max | Percent abs <= 45 | Percent abs <= 90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 41 | 0.223 | 35.426 | -109.301 | -4.811 | 5.330 | 15.626 | 67.511 | 23.916 | 14.503 | 35.523 | 67.511 | 109.301 | 80.488 | 97.561 |
| `1200562` | 13 | 3.059 | 21.407 | -48.705 | 1.407 | 11.624 | 15.527 | 21.588 | 16.415 | 14.503 | 18.592 | 40.796 | 48.705 | 92.308 | 100.000 |
| `1210622` | 14 | 2.739 | 53.756 | -109.301 | -18.401 | 13.871 | 44.744 | 67.511 | 41.576 | 33.769 | 59.285 | 96.091 | 109.301 | 57.143 | 92.857 |
| `1210683` | 14 | -4.927 | 22.484 | -65.188 | -4.540 | 2.280 | 5.259 | 17.428 | 13.221 | 5.187 | 11.881 | 51.415 | 65.188 | 92.857 | 100.000 |
| Summer | 21 | 18.711 | 22.788 | -21.660 | 6.246 | 15.527 | 21.588 | 67.511 | 21.868 | 15.626 | 21.660 | 60.360 | 67.511 | 80.952 | 100.000 |
| Winter | 20 | -19.190 | 36.346 | -109.301 | -43.473 | -0.589 | 5.115 | 19.962 | 26.066 | 10.125 | 43.473 | 89.994 | 109.301 | 80.000 | 95.000 |

Aggregate daily circular differences:

| Scope | n | Signed mean | Signed std | Signed min | Signed q25 | Signed q50 | Signed q75 | Signed max | Abs mean | Abs q50 | Abs q75 | Abs q95 | Abs max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 14 | 3.213 | 26.795 | -64.403 | -1.256 | 9.397 | 19.988 | 36.012 | 19.788 | 15.320 | 22.642 | 50.492 | 64.403 |
| Summer | 7 | 18.373 | 11.829 | -3.146 | 15.320 | 20.789 | 22.158 | 36.012 | 19.272 | 20.789 | 22.158 | 32.146 | 36.012 |
| Winter | 7 | -11.947 | 29.654 | -64.403 | -24.236 | 4.416 | 6.304 | 12.225 | 20.303 | 6.569 | 27.613 | 57.982 | 64.403 |

Direction agreement is usable for diagnostics but not equivalent. The aggregate view reduces the maximum absolute circular difference from `109.301 deg` at station-day level to `64.403 deg`, but aggregate mean absolute circular difference remains about `19.788 deg`.

## Interpretation: schema compatibility, technical data quality, expected source differences, unexplained discrepancies, distribution shift

Schema compatibility:

- The pilot proves the selected ERA5-Land point output can be transformed into a station-day comparison schema aligned with v1 station IDs and dates.
- Required temperature and wind-speed fields are non-null for all 42 paired station-days.
- Direction requires circular handling and explicit calm-threshold nullability.
- The existing v1 station-matrix contract is structurally comparable for the three pilot stations, but ERA5-Land is not a drop-in replacement for the 18-column v1 matrix.

Technical data quality:

- The multi-point validation report passed with no validation issues.
- Hourly, station-day, aggregate, and v1 comparison row counts match the pilot expectations.
- The single direction null is explainable from the documented calm-threshold logic.
- This report used existing pilot outputs only; it did not regenerate data or contact CDS.

Expected source differences:

- ERA5-Land grid cells represent reanalysis model grid points, not necessarily the same physical exposure as v1 station observations.
- Station siting, elevation, local terrain, and model-grid representativeness can cause station-specific temperature and wind-speed offsets.
- ERA5-Land uses hourly UTC component fields; v1 weather files are daily station matrices whose original provider and detailed aggregation method remain unconfirmed.
- Direction comparison is sensitive to calm or variable-wind days, and vector aggregation can differ from any scalar daily direction method used by the original v1 provider.

Unexplained discrepancies:

- Braga Merelim has ERA5-Land wind speed higher than v1 on average (`+0.799 m/s`), while Beja and Guarda are lower (`-0.913 m/s` and `-1.452 m/s`).
- Summer pooled station-day wind-speed correlation is near zero despite stronger within-station correlations.
- Braga Merelim has the largest direction spread, with mean absolute circular difference `41.576 deg` and maximum absolute circular difference `109.301 deg`.
- The pilot does not establish whether these discrepancies come from source definitions, station/grid geography, aggregation differences, short sample windows, or original v1 provider behavior.

Distribution shift:

- Temperature is strongly aligned overall (`Pearson 0.981`, station-day MAD `1.350 deg C`), but station-specific biases are visible.
- Wind speed shows material shift: ERA5-Land is lower than v1 by `0.522 m/s` on average at both station-day and aggregate levels, and station-day MAD is `1.055 m/s`.
- Wind direction differs enough to require explicit circular validation if ERA5-Land becomes a v2 source.
- The observed shifts are large enough that a v2 ERA5-Land weather dataset must be treated as a new data contract, not as a silent extension of v1.

## Model/scaler compatibility implications

The current v1 scalers and trained models must not be claimed valid for ERA5-Land-derived weather.

Compatibility assessment:

- Feature names and high-level concepts can likely be mapped after versioned preprocessing: temperature, wind speed, and wind direction all exist.
- Fitted scaler distributions are not compatible by evidence alone because wind speed and direction distributions shift.
- Current model weights were trained against the v1 weather-source distribution and v1 aggregation choices.
- Replacing v1 weather with ERA5-Land without refitting scalers, retraining models, and re-baselining metrics would create train-serving and historical-distribution risk.

Required future work before any v2 model use:

- Define a versioned ERA5-Land data contract and manifest.
- Extend validators without weakening v1 validation.
- Generate v2 processed features in separate paths.
- Refit scalers on the v2 training distribution.
- Retrain models and compare against v1 baselines.
- Document source selection, rollback, and inference-source compatibility.

## Limitations and deferred questions

Limitations:

- The pilot covers only three mapped stations, not all 17 currently mapped IPMA station IDs or all 18 v1 weather columns.
- The pilot covers only two seven-day windows in 2023.
- No full-year seasonality, weather extremes, long-term drift, or production-target relationship was evaluated.
- The original v1 weather provider, station metadata history, and exact v1 daily aggregation method remain unconfirmed.
- The unmatched v1 station `1200579` remains unresolved.
- This report does not evaluate full-grid ERA5-Land strategies, capacity-weighted wind-farm regions, or WeatherAPI alternatives.
- Direction analysis intentionally uses circular differences; raw-degree correlations are not reported as primary evidence.

Deferred questions:

- Are the station-specific wind-speed biases stable across full years?
- Would all 17 mapped station coordinates improve or worsen the aggregate distribution shift?
- Should a future v2 source use station points, a regular grid, or installed-capacity-weighted regions?
- How should calm and near-calm direction nullability be represented in a future canonical schema?
- What production-source overlap and revision policy will be paired with any v2 weather source?
- How large is the downstream forecast-performance impact after v2 scaler refitting and model retraining?

## Checkpoint status

Checkpoint 3 documentation status: complete for the approved scope.

Safety status:

- Only a documentation comparison was produced.
- No v1 data, processed data, models, scalers, scripts, notebooks, or pilot artifacts were modified.
- No live CDS/network request, notebook execution, training run, pipeline run, or generated-output write was performed.
- No final v2 source was selected.
- Step 2B remains paused and Phase 3 was not started.

Conclusion:

ERA5-Land is technically usable for a versioned v2 weather-source pilot path, but the observed wind-speed and wind-direction shifts mean it is not compatible with the current v1 scalers or trained models as a silent replacement. Any adoption must proceed through versioned ingestion, validation, preprocessing, scaler refitting, model retraining, and metric re-baselining.
