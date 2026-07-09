import pandas as pd

from wind_forecast import schemas


def test_legacy_column_to_english_maps_base_and_pattern_columns():
    assert schemas.legacy_column_to_english(schemas.LEGACY_DATE_COLUMN) == schemas.DATE_COLUMN
    assert (
        schemas.legacy_column_to_english(f"{schemas.RAW_WIND_PRODUCTION_COLUMN}_lag7")
        == "Wind_Production_Lag7"
    )
    assert (
        schemas.legacy_column_to_english(f"{schemas.LEGACY_WIND_SPEED_COLUMN}_roll_std_14")
        == "Average_Wind_Speed_Rolling_Std_14"
    )
    assert schemas.legacy_column_to_english("Unmapped_Column") == "Unmapped_Column"


def test_english_column_to_legacy_maps_base_and_pattern_columns():
    assert schemas.english_column_to_legacy(schemas.DATE_COLUMN) == schemas.LEGACY_DATE_COLUMN
    assert (
        schemas.english_column_to_legacy("Wind_Production_Rolling_Mean_7")
        == f"{schemas.LEGACY_WIND_ROLLING_PREFIX}_roll_mean_7"
    )
    assert (
        schemas.english_column_to_legacy("Average_Temperature_Lag3")
        == f"{schemas.LEGACY_TEMPERATURE_COLUMN}_lag3"
    )
    assert schemas.english_column_to_legacy("Unmapped_Column") == "Unmapped_Column"


def test_columns_to_english_preserves_order():
    columns = [
        schemas.LEGACY_DATE_COLUMN,
        schemas.LEGACY_WIND_SPEED_COLUMN,
        "Unmapped_Column",
    ]

    assert schemas.columns_to_english(columns) == [
        schemas.DATE_COLUMN,
        schemas.AVG_WIND_SPEED_COLUMN,
        "Unmapped_Column",
    ]


def test_rename_legacy_columns_to_english_returns_copy_with_expected_names():
    source = pd.DataFrame(
        {
            schemas.LEGACY_DATE_COLUMN: ["2026-01-01"],
            schemas.LEGACY_WIND_SPEED_COLUMN: [4.5],
        }
    )

    renamed = schemas.rename_legacy_columns_to_english(source)

    assert renamed.columns.tolist() == [schemas.DATE_COLUMN, schemas.AVG_WIND_SPEED_COLUMN]
    assert source.columns.tolist() == [schemas.LEGACY_DATE_COLUMN, schemas.LEGACY_WIND_SPEED_COLUMN]
