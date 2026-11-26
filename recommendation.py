"""
recommendation.py
=================
A simple, data-driven "which crops have historically done well here?" helper.

Method (kept deliberately simple and honest):
  For each crop grown in the chosen place we compare
      local median yield  /  all-India median yield of the SAME crop
  The ratio is called the Performance Index. A value of 1.2 means the crop's typical
  yield here is 20% higher than its typical yield across India. Because it is a ratio
  for the same crop, it is not distorted by crops that use different units.

This is a summary of HISTORICAL records only. It is not agronomic advice: it knows
nothing about soil, rainfall, prices or water availability.
"""

import pandas as pd


def recommend_crops(df, state, district=None, season=None, top_n=5, min_records=5):
    """
    Return the crops with the best historical Performance Index for a place.

    Columns returned: Crop, Records, Local_Median_Yield, National_Median_Yield,
                      Performance_Index
    """
    if state not in set(df["State"]):
        raise ValueError(f"Unknown state: {state}")

    national_median = df.groupby("Crop")["Yield"].median().rename("National_Median_Yield")

    local = df[df["State"] == state]
    if district:
        local = local[local["District"] == district]
    if season:
        local = local[local["Season"] == season]

    summary = (
        local.groupby("Crop")
        .agg(Records=("Yield", "size"), Local_Median_Yield=("Yield", "median"))
        .query("Records >= @min_records")
        .join(national_median)
    )
    summary = summary[summary["National_Median_Yield"] > 0]
    summary["Performance_Index"] = summary["Local_Median_Yield"] / summary["National_Median_Yield"]

    result = (
        summary.sort_values("Performance_Index", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return result[["Crop", "Records", "Local_Median_Yield", "National_Median_Yield", "Performance_Index"]]
