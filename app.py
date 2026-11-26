"""
app.py
======
Streamlit web application for the AI-Based Crop Yield Prediction System.

Run with:   streamlit run app.py

Pages (sidebar navigation):
  Home | Crop Yield Prediction | Crop Recommendation | Data Analytics |
  Model Performance | Prediction History

Before running, create the data and model files:
  python data_preprocessing.py
  python train_model.py
"""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_preprocessing as dp
import database as db
import prediction as pred
import recommendation as rec

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")

GREEN = "#2E7D32"
LIGHT_GREEN = "#A5D6A7"
GREEN_SEQUENCE = ["#2E7D32", "#66BB6A", "#F9A825", "#8D6E63", "#00897B", "#C0CA33", "#6D4C41", "#26A69A"]

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(135deg, #1B5E20 0%, #43A047 100%); padding: 1.8rem 2rem;
       border-radius: 14px; margin-bottom: 1.2rem;}
.hero h1 {color: #FFFFFF; margin: 0 0 .4rem 0; font-size: 2rem;}
.hero p {color: #E8F5E9; margin: 0; font-size: 1.05rem;}
.info-card {background: #F1F8E9; border-left: 6px solid #2E7D32; padding: .9rem 1.2rem;
            border-radius: 8px; color: #1B1B1B; margin-bottom: .8rem;}
.warn-card {background: #FFF8E1; border-left: 6px solid #F9A825; padding: .9rem 1.2rem;
            border-radius: 8px; color: #1B1B1B; margin-bottom: .8rem;}
.result-card {background: #FFFFFF; border: 1px solid #C8E6C9; border-top: 6px solid #2E7D32;
              border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center;
              box-shadow: 0 2px 6px rgba(0,0,0,.06);}
.result-card .label {color: #558B2F; font-size: .85rem; font-weight: 700;
                     text-transform: uppercase; letter-spacing: .05em;}
.result-card .value {color: #1B5E20; font-size: 2.2rem; font-weight: 700; margin: .3rem 0;}
.result-card .unit {color: #546E7A; font-size: .9rem;}
</style>
"""

UNIT_NOTE = (
    "Area is in hectares and production in the units of the source dataset (mostly tonnes). "
    "Some crops, for example coconut, are reported in other units (such as nuts), so compare "
    "yields between crops with care."
)


# ==========================================================================
# Cached loaders
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_data():
    return dp.load_clean_data()


@st.cache_data(show_spinner=False)
def load_options():
    return pred.get_options(load_data())


@st.cache_resource(show_spinner=False)
def get_model():
    return pred.load_model()


@st.cache_data(show_spinner=False)
def load_comparison():
    path = dp.MODELS_DIR / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run:  python train_model.py")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metadata():
    path = dp.MODELS_DIR / "model_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_test_predictions():
    path = dp.MODELS_DIR / "test_predictions.csv"
    return pd.read_csv(path) if path.exists() else None


# ==========================================================================
# Small UI helpers
# ==========================================================================
def info_box(text, kind="info"):
    css_class = "warn-card" if kind == "warn" else "info-card"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def result_card(label, value, unit):
    st.markdown(
        f'<div class="result-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="unit">{unit}</div></div>',
        unsafe_allow_html=True,
    )


def style_figure(fig, height=450):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="#FFFFFF")
    return fig


# ==========================================================================
# Page 1: Home
# ==========================================================================
def page_home(df):
    st.markdown(
        '<div class="hero"><h1>🌾 AI-Based Crop Yield Prediction System</h1>'
        "<p>Machine Learning on historical Indian district-wise crop production data</p></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Introduction")
    st.write(
        "Crop yield is the amount of crop harvested from a unit of land (for example tonnes per hectare). "
        "This system learns patterns from historical records - state, district, crop, season, year and "
        "cultivated area - and uses them to estimate the yield for a new combination. Multiplying the "
        "predicted yield by the cultivated area gives an estimated total production."
    )

    st.subheader("Project objectives")
    st.markdown(
        "- Clean and analyse a real district-wise crop production dataset.\n"
        "- Train and compare Linear Regression, Random Forest and Support Vector Regression.\n"
        "- Evaluate models with MAE, MSE, RMSE and R² on years the models have never seen.\n"
        "- Provide an easy web interface for yield estimates, analytics and crop insights."
    )

    st.subheader("Dataset overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Crops", df["Crop"].nunique())
    c3.metric("States / UTs", df["State"].nunique())
    c4.metric("Districts", df["District"].nunique())
    c5.metric("Years covered", f"{df['Year'].min()}-{df['Year'].max()}")

    columns_table = pd.DataFrame({
        "Column": ["State", "District", "Crop", "Season", "Year", "Area", "Production", "Yield (target)"],
        "Meaning": [
            "State or Union Territory", "District name", "Crop name",
            "Cropping season (e.g. Kharif, Rabi)", "Calendar year of the record",
            "Cultivated area (hectares)", "Total production (used only to compute Yield)",
            "Production / Area - the value the model predicts",
        ],
    })
    st.dataframe(columns_table, hide_index=True)

    st.subheader("Key project information")
    meta = load_metadata()
    k1, k2 = st.columns(2)
    with k1:
        info_box(
            "<b>Model inputs:</b> State, District, Crop, Season, Year, Area<br>"
            "<b>Target:</b> Yield = Production / Area<br>"
            "<b>No data leakage:</b> Production is never used as an input."
        )
    with k2:
        if meta:
            info_box(
                f"<b>Best model:</b> {meta.get('best_model', '-')}<br>"
                f"<b>Test years (unseen):</b> {meta.get('split', {}).get('test_years', '-')}<br>"
                "<b>Split:</b> chronological - train on earlier years, test on the latest."
            )
        else:
            info_box("Model not trained yet. Run <code>python train_model.py</code>.", kind="warn")

    info_box(
        "<b>Important limitation:</b> the dataset has no rainfall, temperature, soil quality or fertilizer "
        "information, so the model cannot account for them. Predictions are statistical estimates based on "
        "historical patterns - <b>not guaranteed harvest results</b>.",
        kind="warn",
    )


# ==========================================================================
# Page 2: Crop Yield Prediction
# ==========================================================================
def page_prediction(df):
    st.title("🌱 Crop Yield Prediction")
    st.caption("Choose the details below and click Predict. Estimates are based on historical records.")

    try:
        model = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # corrupt file, version mismatch ...
        st.error(f"The saved model could not be loaded: {exc}\nTry re-running: python train_model.py")
        st.stop()

    options = load_options()
    meta = load_metadata()

    c1, c2 = st.columns(2)
    with c1:
        state = st.selectbox("State", options["states"])
        district = st.selectbox("District", options["districts_by_state"][state])
        only_local = st.checkbox("Show only crops recorded in this district", value=True)
    local_crops = sorted(df[(df["State"] == state) & (df["District"] == district)]["Crop"].unique())
    crop_choices = local_crops if (only_local and local_crops) else options["crops"]
    with c2:
        crop = st.selectbox("Crop", crop_choices)
        season = st.selectbox("Season", options["seasons"])

    c3, c4 = st.columns(2)
    with c3:
        year = st.number_input(
            "Year", min_value=options["min_year"], max_value=options["max_year"] + 10,
            value=options["max_year"] + 1, step=1,
        )
    with c4:
        area = st.number_input("Cultivated area (hectares)", min_value=0.01, value=100.0, step=10.0)

    if not st.button("🔍 Predict Yield", type="primary"):
        return

    try:
        result = pred.predict_yield(model, state, district, crop, season, int(year), float(area))
    except ValueError as exc:
        st.error(str(exc))
        return

    details = pred.get_crop_details(df, crop, state, district)
    known = pred.is_known_combination(df, state, district, crop, season)

    st.divider()
    st.subheader("Prediction result")
    r1, r2 = st.columns(2)
    with r1:
        result_card("Predicted yield", f"{result['yield_per_unit_area']:,.2f}", "per hectare")
    with r2:
        result_card("Estimated total production", f"{result['estimated_production']:,.2f}",
                    f"= predicted yield x {area:,.2f} hectares")

    if not known:
        st.warning("This exact state + district + crop + season combination was never recorded, "
                   "so the estimate is less reliable.")
    if int(year) > options["max_year"]:
        st.info(f"The data ends in {options['max_year']}. Predictions for later years cannot "
                "capture new trends.")

    tab_details, tab_explain = st.tabs(["📋 Selected crop details", "💡 Explanation"])
    with tab_details:
        st.markdown(f"**{crop}** - grown in {details['states_grown']} states in this dataset.")
        rows = []
        for label, key in [("All India", "national"), (f"{state}", "state"), (f"{district}", "district")]:
            s = details[key]
            if s:
                rows.append({
                    "Area": label, "Records": s["records"], "Median yield": round(s["median_yield"], 2),
                    "Typical range (P10-P90)": f"{s['p10']:,.2f} - {s['p90']:,.2f}",
                    "Years": f"{s['first_year']}-{s['last_year']}",
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        seasons_text = ", ".join(f"{k} ({v:,})" for k, v in details["seasons"].items())
        st.markdown(f"**Seasons recorded (records):** {seasons_text}")
        top = ", ".join(f"{k} ({v:,.2f})" for k, v in details["top_states_by_median_yield"].items())
        st.markdown(f"**States with the highest median yield:** {top}")
        st.caption(UNIT_NOTE)
    with tab_explain:
        st.markdown(pred.explain_prediction(
            result, details, state, district, season, int(year), float(area), known, options["max_year"]))

    try:
        db.save_prediction({
            "state": state, "district": district, "crop": crop, "season": season,
            "year": int(year), "area": float(area),
            "predicted_yield": result["yield_per_unit_area"],
            "estimated_production": result["estimated_production"],
            "model_name": meta.get("best_model", "best_model"),
        })
    except Exception:
        st.caption("(Prediction could not be saved to history.)")


# ==========================================================================
# Page 3: Crop Recommendation
# ==========================================================================
def page_recommendation(df):
    st.title("🌿 Crop Insights")
    st.caption("Which crops have historically performed best in a place, compared with the rest of India?")
    options = load_options()

    c1, c2, c3 = st.columns(3)
    state = c1.selectbox("State", options["states"])
    district = c2.selectbox("District", ["All districts"] + options["districts_by_state"][state])
    season = c3.selectbox("Season", ["All seasons"] + options["seasons"])
    c4, c5 = st.columns(2)
    top_n = c4.slider("Number of crops to show", 3, 15, 5)
    min_records = c5.slider("Minimum records per crop", 1, 30, 5)

    result = rec.recommend_crops(
        df, state,
        district=None if district == "All districts" else district,
        season=None if season == "All seasons" else season,
        top_n=top_n, min_records=min_records,
    )
    if result.empty:
        st.warning("No crop has enough records for this selection. Lower the minimum records or widen the selection.")
        return

    fig = px.bar(result.sort_values("Performance_Index"), x="Performance_Index", y="Crop", orientation="h",
                 text_auto=".2f", title="Performance Index (1.0 = same as the all-India median)")
    fig.update_traces(marker_color=GREEN)
    fig.add_vline(x=1.0, line_dash="dash", line_color="#8D6E63")
    st.plotly_chart(style_figure(fig, 120 + 45 * len(result)))
    st.dataframe(result.round(3), hide_index=True)
    info_box(
        "<b>How to read this:</b> Performance Index = median yield of the crop here / median yield of the "
        "same crop across India. It summarises <b>historical records only</b> and knows nothing about soil, "
        "rainfall, water, costs or market prices, so it is not farming advice.",
        kind="warn",
    )


# ==========================================================================
# Page 4: Data Analytics
# ==========================================================================
def page_analytics(df):
    st.title("📊 Data Analytics")
    st.caption("Interactive exploration of the cleaned dataset.")

    all_crops = sorted(df["Crop"].unique())
    default_excluded = [c for c in all_crops if "coconut" in c.lower()]
    with st.expander("Filters", expanded=True):
        f1, f2 = st.columns([2, 1])
        excluded = f1.multiselect(
            "Exclude crops from the charts", all_crops, default=default_excluded,
            help="Production of different crops is measured in different units, so a few crops can "
                 "dominate the charts. Coconut is excluded by default for this reason.")
        y_min, y_max = int(df["Year"].min()), int(df["Year"].max())
        year_range = f2.slider("Year range", y_min, y_max, (y_min, y_max)) if y_min < y_max else (y_min, y_max)

    fdf = df[(df["Year"].between(*year_range)) & (~df["Crop"].isin(excluded))]
    if fdf.empty:
        st.warning("No data left after filtering. Change the filters.")
        return
    st.caption(UNIT_NOTE)

    tabs = st.tabs(["Crop-wise", "State-wise", "Yearly trend", "Seasons", "Top 10 crops", "Yield", "Area vs Production"])

    # ---- Crop-wise production comparison ----
    with tabs[0]:
        n = st.slider("Number of crops", 5, 40, 15, key="crop_n")
        data = fdf.groupby("Crop", as_index=False)["Production"].sum().nlargest(n, "Production")
        fig = px.bar(data.sort_values("Production"), x="Production", y="Crop", orientation="h",
                     title=f"Total production of the top {n} crops")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(style_figure(fig, max(400, 28 * n + 100)))

    # ---- State-wise crop production ----
    with tabs[1]:
        crop_choice = st.selectbox("Crop", ["All crops"] + sorted(fdf["Crop"].unique()), key="state_crop")
        sub = fdf if crop_choice == "All crops" else fdf[fdf["Crop"] == crop_choice]
        data = sub.groupby("State", as_index=False)["Production"].sum().sort_values("Production")
        fig = px.bar(data, x="Production", y="State", orientation="h",
                     title=f"State-wise production - {crop_choice}")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(style_figure(fig, max(400, 24 * len(data) + 100)))

    # ---- Year-wise production trend ----
    with tabs[2]:
        crop_choice = st.selectbox("Crop", ["All crops"] + sorted(fdf["Crop"].unique()), key="year_crop")
        sub = fdf if crop_choice == "All crops" else fdf[fdf["Crop"] == crop_choice]
        trend = sub.groupby("Year", as_index=False).agg(Production=("Production", "sum"),
                                                        Median_Yield=("Yield", "median"))
        fig = px.line(trend, x="Year", y="Production", markers=True,
                      title=f"Total production by year - {crop_choice}")
        fig.update_traces(line_color=GREEN)
        st.plotly_chart(style_figure(fig))
        if crop_choice != "All crops":
            fig2 = px.line(trend, x="Year", y="Median_Yield", markers=True,
                           title=f"Median yield by year - {crop_choice}")
            fig2.update_traces(line_color="#F9A825")
            st.plotly_chart(style_figure(fig2, 350))
        st.caption("A change in the number of records per year can also change totals, "
                   "not only real production changes.")

    # ---- Season-wise crop production ----
    with tabs[3]:
        data = fdf.groupby("Season", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
        a, b = st.columns(2)
        fig = px.bar(data, x="Season", y="Production", title="Production by season")
        fig.update_traces(marker_color=GREEN)
        a.plotly_chart(style_figure(fig))
        fig = px.pie(data, names="Season", values="Production", hole=0.45, title="Share of production",
                     color_discrete_sequence=GREEN_SEQUENCE)
        b.plotly_chart(style_figure(fig))

    # ---- Top 10 crops by production ----
    with tabs[4]:
        data = fdf.groupby("Crop", as_index=False)["Production"].sum().nlargest(10, "Production")
        fig = px.bar(data.sort_values("Production"), x="Production", y="Crop", orientation="h",
                     text_auto=".3s", title="Top 10 crops by total production")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(style_figure(fig))

    # ---- Crop yield comparison ----
    with tabs[5]:
        n = st.slider("Number of crops", 5, 40, 15, key="yield_n")
        data = fdf.groupby("Crop", as_index=False).agg(Median_Yield=("Yield", "median"),
                                                       Records=("Yield", "size"))
        data = data[data["Records"] >= 20].nlargest(n, "Median_Yield")
        fig = px.bar(data.sort_values("Median_Yield"), x="Median_Yield", y="Crop", orientation="h",
                     title=f"Median yield of {len(data)} crops (crops with at least 20 records)")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(style_figure(fig, max(400, 28 * len(data) + 100)))
        st.caption("Crops use different units, so this chart compares crops only roughly. "
                   "Compare the same crop across places or years for fair conclusions.")

    # ---- Area versus production ----
    with tabs[6]:
        by_count = fdf["Crop"].value_counts().index.tolist()
        crop_choice = st.selectbox("Crop", by_count, key="scatter_crop")
        sub = fdf[fdf["Crop"] == crop_choice]
        sample = sub.sample(n=min(4000, len(sub)), random_state=42)
        fig = px.scatter(sample, x="Area", y="Production", color="Season", log_x=True, log_y=True,
                         opacity=0.6, hover_data=["State", "District", "Year"],
                         color_discrete_sequence=GREEN_SEQUENCE,
                         title=f"Area vs production - {crop_choice} (log scales)")
        st.plotly_chart(style_figure(fig, 500))
        if len(sub) > 2:
            r = np.corrcoef(np.log10(sub["Area"]), np.log10(sub["Production"]))[0, 1]
            st.caption(f"Correlation between log(Area) and log(Production): {r:.3f}. "
                       "Larger areas generally produce more, but yield per hectare varies.")


# ==========================================================================
# Page 5: Model Performance
# ==========================================================================
def page_performance(df):
    st.title("🏆 Model Performance")

    try:
        comparison = load_comparison()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    meta = load_metadata()

    comparison = comparison.sort_values("RMSE").reset_index(drop=True)
    best = comparison.iloc[0]
    best_name = meta.get("best_model", best["Model"])

    c1, c2, c3 = st.columns(3)
    with c1:
        result_card("Best model", best_name, "lowest RMSE on unseen test years")
    with c2:
        result_card("Best model R²", f"{best['R2']:.3f}", "on the test years")
    with c3:
        result_card("Best model RMSE", f"{best['RMSE']:.2f}", "same units as yield")

    split = meta.get("split", {})
    if split:
        st.markdown(
            f"**Train rows:** {split.get('n_train', 0):,} (years {split.get('train_year_min')}-"
            f"{split.get('train_year_max')})  |  **Test rows:** {split.get('n_test', 0):,} "
            f"(years {split.get('test_years')})  |  **Split:** {split.get('strategy')}"
        )

    st.subheader("Comparison table")
    table = comparison.rename(columns={"R2": "R²"})
    st.dataframe(table.style.format({"MAE": "{:.3f}", "MSE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.4f}"}),
                 hide_index=True)

    st.subheader("Comparison charts")
    colors = {m: (GREEN if m == best_name else LIGHT_GREEN) for m in comparison["Model"]}
    cols = st.columns(2)
    for i, (metric, label) in enumerate([("MAE", "MAE (lower is better)"), ("RMSE", "RMSE (lower is better)"),
                                         ("MSE", "MSE (lower is better)"), ("R2", "R² (higher is better)")]):
        fig = px.bar(comparison, x="Model", y=metric, color="Model", color_discrete_map=colors,
                     text_auto=".3f", title=label)
        fig.update_layout(showlegend=False)
        cols[i % 2].plotly_chart(style_figure(fig, 340))

    st.subheader("Why this model was selected")
    st.markdown(
        f"**{best_name}** was selected because it had the **lowest RMSE ({best['RMSE']:.3f})** and an "
        f"R² of **{best['R2']:.3f}** on test years that no model saw during training. Selecting on "
        "future years rather than random rows gives a more honest estimate of real forecasting ability."
    )
    reasons = {
        "Random Forest": "A Random Forest averages many decision trees. Trees can model non-linear effects and "
                         "interactions (for example the same crop behaving differently in different districts) "
                         "without assuming a straight-line relationship.",
        "Linear Regression": "Linear Regression is simple and fast and, with one-hot encoded categories, learns an "
                             "additive effect for every state, crop, season and location. It performs well when "
                             "those effects are mostly additive.",
        "SVR": "SVR with an RBF kernel can fit smooth non-linear patterns. It is slow on large data, so here it "
               f"was trained on a sample of {meta.get('svr_sample_size', 'limited')} rows.",
    }
    if best_name in reasons:
        st.markdown(f"**A likely reason:** {reasons[best_name]}")
    st.markdown(
        "- **Linear Regression:** one weight per feature value; cannot capture interactions by itself.\n"
        "- **Random Forest:** many trees voting together; handles non-linear patterns; larger model file.\n"
        "- **SVR:** flexible but slow, so trained on a sample; sensitive to its settings."
    )
    st.caption("These are general properties of the algorithms. The table above shows how each one actually "
               "behaved on this dataset. Errors are measured on the original yield scale; models learn "
               "log(1 + yield) internally because yields are very skewed.")

    preds = load_test_predictions()
    if preds is not None and not preds.empty:
        st.subheader("Actual vs predicted yield (best model, test years)")
        sample = preds[(preds["Actual_Yield"] > 0) & (preds["Predicted_Yield"] > 0)]
        fig = px.scatter(sample, x="Actual_Yield", y="Predicted_Yield", color="Crop", opacity=0.6,
                         log_x=True, log_y=True, hover_data=["State", "District", "Year"])
        lo = float(min(sample["Actual_Yield"].min(), sample["Predicted_Yield"].min()))
        hi = float(max(sample["Actual_Yield"].max(), sample["Predicted_Yield"].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                 line=dict(color="#8D6E63", dash="dash")))
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_figure(fig, 500))
        st.caption("Points close to the dashed line are accurate predictions (log scales, random sample of test rows).")


# ==========================================================================
# Page 6: Prediction History
# ==========================================================================
def page_history(df):
    st.title("🗂️ Prediction History")
    try:
        history = db.fetch_history()
    except Exception as exc:
        st.error(f"Could not read the history database: {exc}")
        return
    if history.empty:
        st.info("No predictions yet. Make one on the 'Crop Yield Prediction' page.")
        return
    st.dataframe(history, hide_index=True)
    st.download_button("⬇️ Download as CSV", history.to_csv(index=False).encode("utf-8"),
                       file_name="prediction_history.csv", mime="text/csv")
    if st.checkbox("I want to delete all saved predictions"):
        if st.button("Delete history"):
            db.clear_history()
            st.rerun()


# ==========================================================================
# Main
# ==========================================================================
PAGES = {
    "🏠 Home": page_home,
    "🌱 Crop Yield Prediction": page_prediction,
    "🌿 Crop Insights": page_recommendation,
    "📊 Data Analytics": page_analytics,
    "🏆 Model Performance": page_performance,
    "🗂️ Prediction History": page_history,
}


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown("## 🌾 Crop Yield AI")
    page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Educational Machine Learning project. Estimates are not guaranteed harvest results.")

    try:
        df = load_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.info("Setup order:\n\n1. Put `crop_production.csv` in the `dataset` folder\n"
                "2. `python data_preprocessing.py`\n3. `python train_model.py`\n4. `streamlit run app.py`")
        st.stop()

    PAGES[page](df)


main()
