"""
prediction.py
=============
Helper functions used by the Streamlit app (and by test_prediction.py):

  * load_model()          - loads models/best_model.pkl
  * get_options()         - dropdown values (states, districts, crops, seasons ...)
  * predict_yield()       - predicts yield and estimated total production
  * get_crop_details()    - historical statistics for the selected crop
  * is_known_combination() - was this state/district/crop/season ever seen in the data?
  * explain_prediction()  - plain-English explanation of a prediction
"""

import joblib
import numpy as np
import pandas as pd

import data_preprocessing as dp

MODEL_PATH = dp.MODELS_DIR / "best_model.pkl"
FEATURES = dp.FEATURE_COLUMNS


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_model(path=MODEL_PATH):
    """Load the saved model pipeline. Raises a friendly error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}\n"
            "Train it first with:  python train_model.py"
        )
    return joblib.load(path)


def get_options(df):
    """Values for the dropdown menus in the prediction form."""
    states = sorted(df["State"].unique())
    return {
        "states": states,
        "districts_by_state": {
            s: sorted(df.loc[df["State"] == s, "District"].unique()) for s in states
        },
        "crops": sorted(df["Crop"].unique()),
        "seasons": df["Season"].value_counts().index.tolist(),  # most common first
        "min_year": int(df["Year"].min()),
        "max_year": int(df["Year"].max()),
    }


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_yield(model, state, district, crop, season, year, area):
    """
    Predict yield per unit of area and the estimated total production.

    estimated_production = predicted_yield x cultivated_area
    """
    if area is None or not np.isfinite(area) or area <= 0:
        raise ValueError("Cultivated area must be a number greater than 0.")
    if year is None or int(year) != year:
        raise ValueError("Year must be a whole number.")
    for label, value in [("State", state), ("District", district), ("Crop", crop), ("Season", season)]:
        if not str(value).strip():
            raise ValueError(f"{label} must not be empty.")

    row = pd.DataFrame([{
        "State": state, "District": district, "Crop": crop,
        "Season": season, "Year": int(year), "Area": float(area),
    }], columns=FEATURES)

    predicted_yield = float(np.clip(model.predict(row)[0], 0, None))
    return {
        "yield_per_unit_area": predicted_yield,
        "estimated_production": predicted_yield * float(area),
    }


def is_known_combination(df, state, district, crop, season):
    """True if this exact state + district + crop + season appears in the historical data."""
    mask = (
        (df["State"] == state) & (df["District"] == district)
        & (df["Crop"] == crop) & (df["Season"] == season)
    )
    return bool(mask.any())


# --------------------------------------------------------------------------
# Crop details and explanation
# --------------------------------------------------------------------------
def _summary(frame):
    """Summary statistics of yield for a slice of the data (None if the slice is empty)."""
    if frame.empty:
        return None
    return {
        "records": int(len(frame)),
        "median_yield": float(frame["Yield"].median()),
        "mean_yield": float(frame["Yield"].mean()),
        "p10": float(frame["Yield"].quantile(0.10)),
        "p90": float(frame["Yield"].quantile(0.90)),
        "first_year": int(frame["Year"].min()),
        "last_year": int(frame["Year"].max()),
    }


def get_crop_details(df, crop, state=None, district=None):
    """Historical yield statistics for the crop: whole country, state and district."""
    crop_df = df[df["Crop"] == crop]
    state_df = crop_df[crop_df["State"] == state] if state else crop_df.iloc[0:0]
    district_df = (
        state_df[state_df["District"] == district] if (state and district) else crop_df.iloc[0:0]
    )
    top_states = (
        crop_df.groupby("State")["Yield"].median().sort_values(ascending=False).head(3)
    )
    return {
        "crop": crop,
        "national": _summary(crop_df),
        "state": _summary(state_df),
        "district": _summary(district_df),
        "seasons": crop_df["Season"].value_counts().to_dict(),
        "states_grown": int(crop_df["State"].nunique()),
        "top_states_by_median_yield": {k: float(v) for k, v in top_states.items()},
    }


def explain_prediction(result, details, state, district, season, year, area, known_combination, data_max_year):
    """Build a plain-English explanation (Markdown) for one prediction."""
    crop = details["crop"]
    y = result["yield_per_unit_area"]

    # Compare with the most specific history available: district -> state -> country
    reference, level = None, ""
    for key, label in [("district", f"{district} district"), ("state", f"{state} state"), ("national", "all of India")]:
        if details[key]:
            reference, level = details[key], label
            break

    lines = [
        f"The model estimates a yield of **{y:,.2f}** per unit of area for **{crop}** in "
        f"**{district}, {state}** during the **{season}** season of **{year}**. "
        f"Multiplying by your area of {area:,.2f} gives an estimated production of "
        f"**{result['estimated_production']:,.2f}**."
    ]

    if reference:
        median = reference["median_yield"]
        if median > 0:
            change = (y - median) / median * 100
            direction = "above" if change >= 0 else "below"
            lines.append(
                f"For comparison, the typical (median) historical yield of {crop} in {level} is "
                f"**{median:,.2f}** ({reference['records']:,} records, {reference['first_year']}-"
                f"{reference['last_year']}). The prediction is about **{abs(change):.0f}% {direction}** that value."
            )
        lines.append(
            f"Most historical yields in {level} fall between **{reference['p10']:,.2f}** and "
            f"**{reference['p90']:,.2f}** (10th to 90th percentile)."
        )

    lines.append("**How reliable is this?**")
    if not known_combination:
        lines.append(
            "- This exact state + district + crop + season combination never appears in the "
            "historical data, so the estimate relies on similar records and is less reliable."
        )
    if year > data_max_year:
        lines.append(
            f"- The data ends in {data_max_year}. For years after that, models cannot see new "
            "trends (new varieties, policy, climate), so treat the result as a rough guide."
        )
    lines.append(
        "- The model uses only location, crop, season, year and area. The dataset contains no "
        "rainfall, temperature, soil or fertilizer information, so real yields can differ a lot."
    )
    lines.append("- This is a statistical estimate, **not a guaranteed harvest**.")
    return "\n\n".join(lines)
