"""Data preparation step.

Loads the raw E-Commerce churn dataset, cleans it and splits it into a
stratified train / test set.

Usage::

    python -m churn.dataset                # uses data/raw/E Commerce Dataset.xlsx
    python -m churn.dataset --input path/to/file.xlsx
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config

logger = logging.getLogger(__name__)


def find_raw_file(raw_dir: Path = config.RAW_DIR) -> Path:
    """Locate the raw dataset in ``data/raw``.

    Args:
        raw_dir: directory where the raw file is stored.

    Returns:
        Path to the first known raw file found.

    Raises:
        FileNotFoundError: if no raw file is present.
    """
    for name in config.RAW_FILE_CANDIDATES:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No raw dataset found in {raw_dir}. Download 'E Commerce Dataset.xlsx' "
        "from Kaggle and place it in data/raw/."
    )


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw dataset (Excel or CSV).

    The Kaggle Excel file has a ``Data Dict`` sheet and an ``E Comm`` sheet
    with the data; the latter is read.

    Args:
        path: path to an ``.xlsx`` or ``.csv`` file.

    Returns:
        Raw dataframe.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.ExcelFile(path).sheet_names
        sheet = config.RAW_SHEET_NAME if config.RAW_SHEET_NAME in sheets else sheets[-1]
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)
    logger.info("Loaded %s rows x %s columns from %s", *df.shape, path.name)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Check that all expected columns are present.

    Args:
        df: raw dataframe.

    Raises:
        ValueError: if columns are missing.
    """
    expected = (
        [config.ID_COL, config.TARGET] + config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    )
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data.

    * harmonises inconsistent category labels (e.g. ``Phone`` vs
      ``Mobile Phone``),
    * strips whitespace in text columns,
    * removes duplicated customers (same ID, or identical on all other columns),
    * casts the target to ``int``.

    Missing values are *not* imputed here: imputation is learnt on the
    training set inside the model pipeline to avoid data leakage.

    Args:
        df: raw dataframe.

    Returns:
        Cleaned dataframe.
    """
    df = df.copy()
    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace(config.CATEGORY_MAPPING.get(col, {}))
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    n_before = len(df)
    df = df.drop_duplicates(subset=config.ID_COL)
    # Rows identical on every column except the ID are copies of the same
    # customer: if kept, the same record can end up in both train and test
    # and inflate the test metrics (data leakage).
    df = df.drop_duplicates(subset=[c for c in df.columns if c != config.ID_COL])
    if len(df) < n_before:
        logger.info("Dropped %d duplicated customers", n_before - len(df))
    df = df.dropna(subset=[config.TARGET])
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = config.TEST_SIZE):
    """Stratified train / test split.

    Args:
        df: cleaned dataframe.
        test_size: share of rows kept for the test set.

    Returns:
        tuple: ``(train_df, test_df)``.
    """
    return train_test_split(
        df,
        test_size=test_size,
        stratify=df[config.TARGET],
        random_state=config.RANDOM_STATE,
    )


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values and cardinality per column.

    Args:
        df: dataframe to profile.

    Returns:
        A dataframe with dtype, missing count / share and number of unique
        values per column.
    """
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
        }
    )


def run(input_path=None) -> None:
    """Execute the full data preparation step and write output files."""
    path = Path(input_path) if input_path else find_raw_file()
    raw = load_raw(path)
    validate_schema(raw)
    df = clean(raw)

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    data_quality_report(raw).to_csv(config.REPORTS_DIR / "data_quality_report.csv")

    train_df, test_df = split(df)
    logger.info(
        "Clean data: %d rows, churn rate %.2f%% | train %d / test %d",
        len(df),
        df[config.TARGET].mean() * 100,
        len(train_df),
        len(test_df),
    )
    # The interim split is stored before feature engineering
    train_df.to_csv(config.INTERIM_DIR / "train_raw.csv", index=False)
    test_df.to_csv(config.INTERIM_DIR / "test_raw.csv", index=False)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="path to the raw dataset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(args.input)


if __name__ == "__main__":
    main()
