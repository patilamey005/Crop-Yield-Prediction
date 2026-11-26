"""
data_preprocessing.py
=====================
Cleans the raw crop production CSV and runs Exploratory Data Analysis (EDA).

What this file does
-------------------
1. Loads dataset/crop_production.csv
2. Renames columns to: State, District, Crop, Year, Season, Area, Production
3. Handles missing values, duplicates, zero/negative area and invalid production
4. Creates the target column:  Yield = Production / Area
5. Removes extreme yield outliers (checked separately for every crop)
6. Saves dataset/cleaned_crop_data.csv and a small demo sample (demo_data/)
7. Saves EDA charts (PNG) into the plots/ folder

Run it with:   python data_preprocessing.py

Other files import the paths and helper functions defined here.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths (all relative to this file, so the project works from any folder)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
DEMO_DIR = BASE_DIR / "demo_data"

RAW_CSV = DATASET_DIR / "crop_production.csv"
CLEAN_CSV = DATASET_DIR / "cleaned_crop_data.csv"
DEMO_CSV = DEMO_DIR / "sample_crop_data.csv"

# --------------------------------------------------------------------------
# Column settings
# --------------------------------------------------------------------------
# The Kaggle file uses names such as State_Name / District_Name / Crop_Year.
# We accept a few spellings and rename them to one standard set.
COLUMN_ALIASES = {
    "State": ["state_name", "state"],
    "District": ["district_name", "district"],
    "Crop": ["crop"],
    "Year": ["crop_year", "year"],
    "Season": ["season"],
    "Area": ["area"],
    "Production": ["production"],
}
REQUIRED_COLUMNS = ["State", "District", "Crop", "Year", "Season", "Area", "Production"]
TEXT_COLUMNS = ["State", "District", "Crop", "Season"]

# Columns the model is allowed to use as INPUT.
# "Production" is deliberately NOT here: Yield = Production / Area, so using
# Production as an input would leak the answer to the model (data leakage).
FEATURE_COLUMNS = ["State", "District", "Crop", "Season", "Year", "Area"]
TARGET_COLUMN = "Yield"

# --------------------------------------------------------------------------
# Cleaning rules (change these here if your guide asks for different rules)
# --------------------------------------------------------------------------
DROP_ZERO_PRODUCTION = True      # Production = 0 is usually "not reported"
OUTLIER_IQR_FACTOR = 3.0         # a yield beyond Q3 + 3*IQR of its crop is an outlier
MIN_RECORDS_FOR_OUTLIER_CHECK = 20
MIN_VALID_YEAR = 1900
MAX_VALID_YEAR = 2100


# ==========================================================================
# Helper used inside the saved model pipeline
# ==========================================================================
def add_location_feature(X):
    """
    Adds a 'Location' column = State + District.

    Why? Some district names exist in more than one state. Joining the state
    and district name keeps them apart. This function lives in this module (not
    in train_model.py) so that the saved model can be loaded by app.py.
    """
    X = X.copy()
    X["Location"] = X["State"].astype(str) + " | " + X["District"].astype(str)
    return X


# ==========================================================================
# Loading
# ==========================================================================
def standardise_columns(df):
    """Rename columns to the standard names and keep only the required ones."""
    lookup = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            lookup[alias] = standard_name

    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in lookup:
            rename_map[col] = lookup[key]
    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The dataset is missing required column(s): {missing}.\n"
            f"Columns found in your file: {list(df.columns)}"
        )
    return df[REQUIRED_COLUMNS].copy()


def load_raw_data(path=RAW_CSV):
    """Read the raw CSV and standardise its column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download 'Crop Production in India' (crop_production.csv) from Kaggle "
            "and place it inside the 'dataset' folder."
        )
    try:
        df = pd.read_csv(path, keep_default_na=False, na_values=["", "nan", "NaN"])
    except Exception as exc:  # unreadable / corrupt file
        raise ValueError(f"Could not read {path}: {exc}") from exc
    return standardise_columns(df)


def load_clean_data(path=CLEAN_CSV):
    """Read the cleaned CSV created by this file (used by train_model.py and app.py)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}\n"
            "Run this first:  python data_preprocessing.py"
        )
    df = pd.read_csv(path, keep_default_na=False, na_values=["", "nan", "NaN"])
    needed = REQUIRED_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}. Re-run data_preprocessing.py")
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str)
    df["Year"] = df["Year"].astype(int)
    return df


# ==========================================================================
# Cleaning
# ==========================================================================
def remove_yield_outliers(df):
    """
    Remove extreme yields, judged separately for each crop.

    Different crops have very different normal yields (and even different units,
    e.g. coconut), so one global rule would be wrong. For each crop with enough
    records we remove yields above  Q3 + 3 * IQR  (or below Q1 - 3 * IQR).
    """
    grouped = df.groupby("Crop")["Yield"]
    bounds = grouped.quantile([0.25, 0.75]).unstack()
    bounds.columns = ["q1", "q3"]
    bounds["n"] = grouped.size()
    bounds["iqr"] = bounds["q3"] - bounds["q1"]
    bounds["low"] = bounds["q1"] - OUTLIER_IQR_FACTOR * bounds["iqr"]
    bounds["high"] = bounds["q3"] + OUTLIER_IQR_FACTOR * bounds["iqr"]

    joined = df.join(bounds[["n", "iqr", "low", "high"]], on="Crop")
    is_outlier = (
        (joined["n"] >= MIN_RECORDS_FOR_OUTLIER_CHECK)
        & (joined["iqr"] > 0)
        & ((joined["Yield"] < joined["low"]) | (joined["Yield"] > joined["high"]))
    )
    removed_by_crop = df.loc[is_outlier.values, "Crop"].value_counts()
    return df.loc[~is_outlier.values].copy(), removed_by_crop


def clean_data(df):
    """
    Clean the standardised raw data. Returns (clean_df, report_dict).

    Steps: tidy text -> convert numbers -> drop missing -> drop duplicates ->
    drop invalid rows -> compute Yield -> remove outliers.
    """
    report = {"rows_raw": len(df)}
    df = df.copy()

    # 1) Tidy text: the raw file has trailing spaces, e.g. "Kharif     "
    for col in TEXT_COLUMNS:
        text = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        df[col] = text.replace("", pd.NA)

    # 2) Convert numeric columns (bad values become NaN instead of crashing)
    for col in ["Year", "Area", "Production"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) Missing values
    report["missing_per_column"] = {k: int(v) for k, v in df.isna().sum().items()}
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[np.isfinite(df[["Year", "Area", "Production"]]).all(axis=1)]
    report["dropped_missing_values"] = before - len(df)

    # 4) Exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    report["dropped_duplicates"] = before - len(df)

    # 5) Invalid rows
    before = len(df)
    df = df[df["Area"] > 0]
    report["dropped_zero_or_negative_area"] = before - len(df)

    before = len(df)
    if DROP_ZERO_PRODUCTION:
        df = df[df["Production"] > 0]
    else:
        df = df[df["Production"] >= 0]
    report["dropped_invalid_production"] = before - len(df)

    before = len(df)
    df["Year"] = df["Year"].round().astype(int)
    df = df[(df["Year"] >= MIN_VALID_YEAR) & (df["Year"] <= MAX_VALID_YEAR)]
    report["dropped_invalid_year"] = before - len(df)

    # 6) Target column
    df["Yield"] = df["Production"] / df["Area"]
    df = df[np.isfinite(df["Yield"])]

    # 7) Outliers (per crop)
    before = len(df)
    df, removed_by_crop = remove_yield_outliers(df)
    report["dropped_outliers"] = before - len(df)
    report["outliers_by_crop_top5"] = {k: int(v) for k, v in removed_by_crop.head(5).items()}

    # 8) Final types and order
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str)
    df["Area"] = df["Area"].astype(float)
    df["Production"] = df["Production"].astype(float)
    df = df.sort_values(["Year", "State", "District", "Crop", "Season"]).reset_index(drop=True)

    report["rows_clean"] = len(df)
    return df, report


def print_report(report):
    """Print a readable cleaning summary."""
    print("\n=== CLEANING REPORT ===")
    print(f"Rows in raw file                    : {report['rows_raw']:,}")
    print("Missing values per column (raw)     :", report["missing_per_column"])
    print(f"Dropped - missing/invalid values    : {report['dropped_missing_values']:,}")
    print(f"Dropped - duplicate rows            : {report['dropped_duplicates']:,}")
    print(f"Dropped - zero/negative area        : {report['dropped_zero_or_negative_area']:,}")
    print(f"Dropped - zero/negative production  : {report['dropped_invalid_production']:,}")
    print(f"Dropped - invalid year              : {report['dropped_invalid_year']:,}")
    print(f"Dropped - yield outliers (per crop) : {report['dropped_outliers']:,}")
    if report["outliers_by_crop_top5"]:
        print("  Most outliers removed from        :", report["outliers_by_crop_top5"])
    print(f"Rows in cleaned file                : {report['rows_clean']:,}")


def save_clean_data(df, path=CLEAN_CSV):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def make_demo_sample(df, n_rows=2000, path=DEMO_CSV):
    """Save a small random sample of the REAL cleaned data (handy for GitHub / quick tests)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = df.sample(n=min(n_rows, len(df)), random_state=42)
    sample = sample.sort_values(["Year", "State", "District", "Crop"])
    sample.to_csv(path, index=False)
    return path


# ==========================================================================
# Exploratory Data Analysis (Matplotlib + Seaborn, saved as PNG files)
# ==========================================================================
def run_eda(df, out_dir=PLOTS_DIR):
    """Print summary statistics and save EDA charts to the plots/ folder."""
    import matplotlib

    matplotlib.use("Agg")  # draw to files, no pop-up windows
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    green = "#43A047"
    saved = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=120)
        plt.close(fig)
        saved.append(name)

    # ---- text summary ----
    print("\n=== EDA SUMMARY ===")
    print(f"Records: {len(df):,} | States: {df['State'].nunique()} | "
          f"Districts: {df['District'].nunique()} | Crops: {df['Crop'].nunique()}")
    print(f"Years: {df['Year'].min()} to {df['Year'].max()} | Seasons: {sorted(df['Season'].unique())}")
    print("\nNumeric summary:")
    print(df[["Year", "Area", "Production", "Yield"]].describe().round(2).to_string())

    # 1) Top 10 crops by number of records
    counts = df["Crop"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index, color=green, ax=ax)
    ax.set_title("Top 10 crops by number of records")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    save(fig, "01_top_crops_by_records.png")

    # 2) Records per year
    per_year = df["Year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(per_year.index.astype(str), per_year.values, color=green)
    ax.set_title("Number of records per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=60)
    save(fig, "02_records_per_year.png")

    # 3) Records per season
    seasons = df["Season"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(seasons.index, seasons.values, color=green)
    ax.set_title("Number of records per season")
    ax.set_ylabel("Records")
    save(fig, "03_records_per_season.png")

    # 4) Yield distribution (log scale because yield is very skewed)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(np.log1p(df["Yield"]), bins=50, color=green, ax=ax)
    ax.set_title("Distribution of Yield (log scale)")
    ax.set_xlabel("log(1 + Yield)")
    save(fig, "04_yield_distribution.png")

    # 5) Area vs Production (log-log)
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sample["Area"], sample["Production"], s=6, alpha=0.4, color=green)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Area vs Production (log-log, 5000-row sample)")
    ax.set_xlabel("Area")
    ax.set_ylabel("Production")
    save(fig, "05_area_vs_production.png")

    # 6) Correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[["Year", "Area", "Production", "Yield"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greens", ax=ax)
    ax.set_title("Correlation between numeric columns")
    save(fig, "06_correlation_heatmap.png")

    # 7) Yield spread for the 8 most common crops (shows outliers / differences)
    top_crops = df["Crop"].value_counts().head(8).index
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=df[df["Crop"].isin(top_crops)], x="Yield", y="Crop", color="#A5D6A7", ax=ax)
    ax.set_xscale("log")
    ax.set_title("Yield spread for the 8 most common crops (log scale)")
    save(fig, "07_yield_boxplot_top_crops.png")

    print(f"\nSaved {len(saved)} EDA charts to: {out_dir}")
    return saved


# ==========================================================================
# Run as a script
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="Clean the crop dataset and run EDA.")
    parser.add_argument("--no-eda", action="store_true", help="skip saving EDA charts")
    parser.add_argument("--demo-rows", type=int, default=2000, help="rows in demo_data sample")
    args = parser.parse_args()

    try:
        print(f"Reading {RAW_CSV} ...")
        raw = load_raw_data()
        clean, report = clean_data(raw)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print_report(report)
    print(f"\nSaved cleaned data -> {save_clean_data(clean)}")
    print(f"Saved demo sample  -> {make_demo_sample(clean, args.demo_rows)}")

    if not args.no_eda:
        run_eda(clean)
    print("\nDone. Next step:  python train_model.py")


if __name__ == "__main__":
    main()
