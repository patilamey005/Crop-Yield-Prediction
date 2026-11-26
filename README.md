# 🌾 AI-Based Crop Yield Prediction System

A third-year engineering Machine Learning project. It learns from historical Indian district-wise
crop production records and estimates **crop yield** (`Yield = Production / Area`) for a chosen
state, district, crop, season, year and cultivated area. A Streamlit web app lets you predict,
explore the data and compare the models.

> **Important:** the dataset contains **no rainfall, temperature, soil or fertilizer data**. The
> model only sees location, crop, season, year and area, so every result is a *statistical estimate
> from historical patterns*, **not a guaranteed harvest**.

## Project structure

```
Crop_Yield_Prediction/
├── .streamlit/config.toml     # app theme
├── dataset/                   # put crop_production.csv here (cleaned file is created here)
├── demo_data/                 # small sample of your real cleaned data (auto-created)
├── models/                    # best_model.pkl, model_comparison.csv, metadata (auto-created)
├── plots/                     # EDA charts (auto-created)
├── screenshots/               # put your app screenshots here for the report
├── app.py                     # Streamlit web app (6 pages)
├── data_preprocessing.py      # cleaning + Yield column + EDA charts
├── train_model.py             # trains and compares 3 models, saves the best
├── prediction.py              # prediction + crop details + explanation helpers
├── recommendation.py          # "Crop Insights" (historical Performance Index)
├── database.py                # SQLite prediction history
├── test_prediction.py         # automated tests
├── requirements.txt
└── README.md
```

## Setup (Windows + VS Code)

1. Install **Python 3.11 or 3.12** (tick *Add python.exe to PATH*) and **VS Code** with the Python extension.
2. Open this folder in VS Code, then open **Terminal → New Terminal**.
3. Create and activate a virtual environment, then install the libraries:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

## Dataset

Download **"Crop Production in India"** from Kaggle (search for it; uploaded by Abhinand) and copy
`crop_production.csv` into the `dataset/` folder. Check the Kaggle page for the original source and
licence and cite it in your report.

Expected columns: `State_Name, District_Name, Crop_Year, Season, Crop, Area, Production`
(the code renames them to `State, District, Year, Season, Crop, Area, Production`).

## Run order

| Step | Command | What you should see |
|------|---------|---------------------|
| 1 | `python data_preprocessing.py` | Cleaning report, EDA summary, `cleaned_crop_data.csv` and 7 PNG charts in `plots/` |
| 2 | `python train_model.py` | Data split, training log for 3 models, comparison table, best model saved in `models/` |
| 3 | `streamlit run app.py` | Browser opens at `http://localhost:8501` |
| 4 | `python test_prediction.py` | All tests `ok` |

Training is slower on a big dataset. To speed it up: `python train_model.py --rf-trees 30 --svr-sample 5000`.

## How it works

1. **Cleaning** – trims text, removes missing values, exact duplicates, zero/negative area, zero
   production (usually "not reported") and per-crop extreme yield outliers (beyond Q3 + 3×IQR).
2. **Target** – `Yield = Production / Area`. `Production` is **never** a model input (data leakage).
3. **Split** – chronological: earlier years train, the latest ~20% of years test.
4. **Pipeline** – a `Location` (state + district) feature is added because some district names
   repeat across states; categories are one-hot encoded; Year is scaled; Area is log-scaled.
5. **Models** – Linear Regression, Random Forest, SVR (trained on a sample because it is slow).
   Models learn `log(1 + Yield)` because yield is very skewed; metrics are on the original scale.
6. **Selection** – lowest RMSE on the unseen test years. The best pipeline is saved with Joblib.
7. **App** – loads the cleaned data and the saved pipeline.

## App pages

Home · Crop Yield Prediction · Crop Insights · Data Analytics · Model Performance · Prediction History

## Honest notes (read before your viva)

- **No results are included.** All scores come from *your* run of `train_model.py`. Do not copy
  numbers from anywhere else into your report.
- **Units differ between crops.** Area is in hectares; production is mostly in tonnes, but some
  crops (for example coconut) use other units, so yields are not comparable across all crops.
  The Analytics page excludes coconut by default for this reason.
- **Years beyond the data:** tree-based models cannot extrapolate trends, so predictions for years
  after the last year in the data behave much like the last year.
- **Crop Insights** summarises historical records only; it is not farming advice.
- **Possible improvements (Future Scope):** add real weather and soil data, try Gradient Boosting,
  tune hyperparameters with cross-validation, and add prediction intervals.

## Common errors

| Error | Fix |
|-------|-----|
| `python` is not recognized | Reinstall Python and tick *Add python.exe to PATH* |
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `ModuleNotFoundError` | Activate the venv (`venv\Scripts\activate`) and run `pip install -r requirements.txt` |
| `Dataset not found` | Place `crop_production.csv` inside `dataset/` |
| `Cleaned dataset not found` | Run `python data_preprocessing.py` |
| `Trained model not found` | Run `python train_model.py` |
| `streamlit` is not recognized | Use `python -m streamlit run app.py` |
| Model fails to load after upgrading libraries | Re-run `python train_model.py` |
| Training takes very long | Use `--rf-trees 30 --svr-sample 5000` |
| Port already in use | `streamlit run app.py --server.port 8502` |
