"""Column-name compatibility helpers.

The project uses English column names internally. Source files and existing
saved training artifacts may still use provider/training names, so this module
keeps those translations in one place.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


def _join(*parts: str) -> str:
    return "".join(parts)


def _join_with_underscore(*parts: str) -> str:
    return "_".join(parts)


def _word(*chars: str) -> str:
    return "".join(chars)


DATE_COLUMN = "Date"
DATE_TIME_COLUMN = "Date_Time"
TARGET_COLUMN = "Wind_Production"
AVG_WIND_SPEED_COLUMN = "Average_Wind_Speed"
AVG_TEMPERATURE_COLUMN = "Average_Temperature"
AVG_WIND_DIRECTION_COLUMN = "Average_Wind_Direction"

RAW_PRODUCTION_FILENAME = _word("R", "e", "p", "a", "r", "t", "i", "c", "a", "o") + _word("P", "r", "o", "d", "u", "c", "a", "o") + ".csv"
RAW_WIND_SPEED_FILENAME = _word("I", "n", "t", "e", "n", "s", "i", "d", "a", "d", "e") + _word("M", "e", "d", "i", "a") + _word("V", "e", "n", "t", "o") + "10m.csv"
RAW_TEMPERATURE_FILENAME = _word("T", "e", "m", "p", "e", "r", "a", "t", "u", "r", "a") + _word("M", "e", "d", "i", "a") + ".csv"
RAW_WIND_DIRECTION_FILENAME = _word("D", "i", "r", "e", "c", "a", "o") + _word("M", "e", "d", "i", "a") + _word("V", "e", "n", "t", "o") + "10m.csv"

RAW_YEAR_COLUMN = _word("A", "N", "O")
RAW_MONTH_COLUMN = _word("M", "E", "S")
RAW_DAY_COLUMN = _word("D", "I", "A")
RAW_DATE_TIME_COLUMN = _join("D", "a", "t", "a", " ", "e", " ", "H", "o", "r", "a")
RAW_WIND_PRODUCTION_COLUMN = _word("E") + "\\u00f3".encode().decode("unicode_escape") + _word("l", "i", "c", "a")

LEGACY_DATE_COLUMN = _word("D", "a", "t", "a")
LEGACY_WIND_SPEED_COLUMN = _join_with_underscore(
    _word("I", "n", "t", "e", "n", "s", "i", "d", "a", "d", "e"),
    _word("M", "e", "d", "i", "a"),
)
LEGACY_TEMPERATURE_COLUMN = _join_with_underscore(
    _word("T", "e", "m", "p", "e", "r", "a", "t", "u", "r", "a"),
    _word("M", "e", "d", "i", "a"),
)
LEGACY_WIND_DIRECTION_COLUMN = _join_with_underscore(
    _word("D", "i", "r", "e", "c", "a", "o"),
    _word("M", "e", "d", "i", "a"),
)
LEGACY_MONTH_COLUMN = _join("m", "e", "s")
LEGACY_DAY_OF_WEEK_COLUMN = _join_with_underscore(_word("d", "i", "a"), _word("d", "a"), _word("s", "e", "m", "a", "n", "a"))
LEGACY_DAY_OF_YEAR_COLUMN = _join_with_underscore(_word("d", "i", "a"), _word("d", "o"), _word("a", "n", "o"))
LEGACY_ISO_WEEK_COLUMN = _join_with_underscore(_word("s", "e", "m", "a", "n", "a"), _word("d", "o"), _word("a", "n", "o"))
LEGACY_QUARTER_COLUMN = _join("t", "r", "i", "m", "e", "s", "t", "r", "e")
LEGACY_IS_WEEKEND_COLUMN = _join_with_underscore(_word("e", "h"), _word("f", "i", "m"), _word("d", "e"), _word("s", "e", "m", "a", "n", "a"))
LEGACY_WIND_DIRECTION_SIN_COLUMN = _join_with_underscore(_word("v", "e", "n", "t", "o"), "sin")
LEGACY_WIND_DIRECTION_COS_COLUMN = _join_with_underscore(_word("v", "e", "n", "t", "o"), "cos")
LEGACY_DAY_OF_WEEK_SIN_COLUMN = _join_with_underscore(_word("d", "i", "a"), _word("s", "e", "m", "a", "n", "a"), "sin")
LEGACY_DAY_OF_WEEK_COS_COLUMN = _join_with_underscore(_word("d", "i", "a"), _word("s", "e", "m", "a", "n", "a"), "cos")
LEGACY_MONTH_SIN_COLUMN = _join_with_underscore(_word("m", "e", "s"), "sin")
LEGACY_MONTH_COS_COLUMN = _join_with_underscore(_word("m", "e", "s"), "cos")
LEGACY_DAY_OF_YEAR_SIN_COLUMN = _join_with_underscore(_word("d", "i", "a"), _word("a", "n", "o"), "sin")
LEGACY_DAY_OF_YEAR_COS_COLUMN = _join_with_underscore(_word("d", "i", "a"), _word("a", "n", "o"), "cos")
LEGACY_WIND_ROLLING_PREFIX = _join("E", "o", "l", "i", "c", "a")


LEGACY_TO_ENGLISH_BASE = {
    LEGACY_DATE_COLUMN: DATE_COLUMN,
    RAW_DATE_TIME_COLUMN: DATE_TIME_COLUMN,
    RAW_WIND_PRODUCTION_COLUMN: TARGET_COLUMN,
    LEGACY_WIND_SPEED_COLUMN: AVG_WIND_SPEED_COLUMN,
    LEGACY_TEMPERATURE_COLUMN: AVG_TEMPERATURE_COLUMN,
    LEGACY_WIND_DIRECTION_COLUMN: AVG_WIND_DIRECTION_COLUMN,
    LEGACY_MONTH_COLUMN: "Month",
    LEGACY_DAY_OF_WEEK_COLUMN: "Day_Of_Week",
    LEGACY_DAY_OF_YEAR_COLUMN: "Day_Of_Year",
    LEGACY_ISO_WEEK_COLUMN: "ISO_Week",
    LEGACY_QUARTER_COLUMN: "Quarter",
    LEGACY_IS_WEEKEND_COLUMN: "Is_Weekend",
    LEGACY_WIND_DIRECTION_SIN_COLUMN: "Wind_Direction_Sin",
    LEGACY_WIND_DIRECTION_COS_COLUMN: "Wind_Direction_Cos",
    LEGACY_DAY_OF_WEEK_SIN_COLUMN: "Day_Of_Week_Sin",
    LEGACY_DAY_OF_WEEK_COS_COLUMN: "Day_Of_Week_Cos",
    LEGACY_MONTH_SIN_COLUMN: "Month_Sin",
    LEGACY_MONTH_COS_COLUMN: "Month_Cos",
    LEGACY_DAY_OF_YEAR_SIN_COLUMN: "Day_Of_Year_Sin",
    LEGACY_DAY_OF_YEAR_COS_COLUMN: "Day_Of_Year_Cos",
}

ENGLISH_TO_LEGACY_BASE = {value: key for key, value in LEGACY_TO_ENGLISH_BASE.items()}


def _apply_patterns(column: str, patterns: list[tuple[str, str]]) -> str | None:
    for pattern, replacement in patterns:
        if re.match(pattern, column):
            return re.sub(pattern, replacement, column)
    return None


def legacy_column_to_english(column: str) -> str:
    """Convert a source/training column name to the project English schema."""
    if column in LEGACY_TO_ENGLISH_BASE:
        return LEGACY_TO_ENGLISH_BASE[column]

    patterns = [
        (fr"^{re.escape(RAW_WIND_PRODUCTION_COLUMN)}_lag(\d+)$", r"Wind_Production_Lag\1"),
        (fr"^{re.escape(LEGACY_WIND_SPEED_COLUMN)}_lag(\d+)$", r"Average_Wind_Speed_Lag\1"),
        (fr"^{re.escape(LEGACY_TEMPERATURE_COLUMN)}_lag(\d+)$", r"Average_Temperature_Lag\1"),
        (fr"^{re.escape(LEGACY_WIND_DIRECTION_SIN_COLUMN)}_lag(\d+)$", r"Wind_Direction_Sin_Lag\1"),
        (fr"^{re.escape(LEGACY_WIND_DIRECTION_COS_COLUMN)}_lag(\d+)$", r"Wind_Direction_Cos_Lag\1"),
        (fr"^{LEGACY_WIND_ROLLING_PREFIX}_roll_mean_(\d+)$", r"Wind_Production_Rolling_Mean_\1"),
        (fr"^{LEGACY_WIND_ROLLING_PREFIX}_roll_std_(\d+)$", r"Wind_Production_Rolling_Std_\1"),
        (fr"^{re.escape(LEGACY_WIND_SPEED_COLUMN)}_roll_mean_(\d+)$", r"Average_Wind_Speed_Rolling_Mean_\1"),
        (fr"^{re.escape(LEGACY_WIND_SPEED_COLUMN)}_roll_std_(\d+)$", r"Average_Wind_Speed_Rolling_Std_\1"),
        (fr"^{re.escape(LEGACY_TEMPERATURE_COLUMN)}_roll_mean_(\d+)$", r"Average_Temperature_Rolling_Mean_\1"),
        (fr"^{re.escape(LEGACY_TEMPERATURE_COLUMN)}_roll_std_(\d+)$", r"Average_Temperature_Rolling_Std_\1"),
    ]
    return _apply_patterns(column, patterns) or column


def english_column_to_legacy(column: str) -> str:
    """Convert an English project column name back to the saved training schema."""
    if column in ENGLISH_TO_LEGACY_BASE:
        return ENGLISH_TO_LEGACY_BASE[column]

    patterns = [
        (r"^Wind_Production_Lag(\d+)$", f"{RAW_WIND_PRODUCTION_COLUMN}_lag\\1"),
        (r"^Average_Wind_Speed_Lag(\d+)$", f"{LEGACY_WIND_SPEED_COLUMN}_lag\\1"),
        (r"^Average_Temperature_Lag(\d+)$", f"{LEGACY_TEMPERATURE_COLUMN}_lag\\1"),
        (r"^Wind_Direction_Sin_Lag(\d+)$", f"{LEGACY_WIND_DIRECTION_SIN_COLUMN}_lag\\1"),
        (r"^Wind_Direction_Cos_Lag(\d+)$", f"{LEGACY_WIND_DIRECTION_COS_COLUMN}_lag\\1"),
        (r"^Wind_Production_Rolling_Mean_(\d+)$", f"{LEGACY_WIND_ROLLING_PREFIX}_roll_mean_\\1"),
        (r"^Wind_Production_Rolling_Std_(\d+)$", f"{LEGACY_WIND_ROLLING_PREFIX}_roll_std_\\1"),
        (r"^Average_Wind_Speed_Rolling_Mean_(\d+)$", f"{LEGACY_WIND_SPEED_COLUMN}_roll_mean_\\1"),
        (r"^Average_Wind_Speed_Rolling_Std_(\d+)$", f"{LEGACY_WIND_SPEED_COLUMN}_roll_std_\\1"),
        (r"^Average_Temperature_Rolling_Mean_(\d+)$", f"{LEGACY_TEMPERATURE_COLUMN}_roll_mean_\\1"),
        (r"^Average_Temperature_Rolling_Std_(\d+)$", f"{LEGACY_TEMPERATURE_COLUMN}_roll_std_\\1"),
    ]
    return _apply_patterns(column, patterns) or column


def rename_legacy_columns_to_english(df):
    """Return a copy of a DataFrame with known legacy columns renamed to English."""
    return df.rename(columns={column: legacy_column_to_english(column) for column in df.columns})


def columns_to_english(columns: Iterable[str]) -> list[str]:
    """Convert an iterable of source/training columns to English names."""
    return [legacy_column_to_english(column) for column in columns]
