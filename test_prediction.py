"""
test_prediction.py
==================
Automated checks for the project. Run with:

    python test_prediction.py

Tests that need the trained model / cleaned data are skipped (with a message)
until you have run data_preprocessing.py and train_model.py.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import data_preprocessing as dp
import database as db
import prediction as pred
import recommendation as rec
import train_model as tm

HAS_MODEL = pred.MODEL_PATH.exists() and dp.CLEAN_CSV.exists()


def make_raw_sample():
    """A tiny hand-made raw table that contains every kind of dirty row."""
    return pd.DataFrame({
        "State_Name": ["A", "A", "A", "A", "A", "A", "B"],
        "District_Name": ["D1", "D1", "D1", "D1", "D2", "D2", "D3"],
        "Crop_Year": [2000, 2000, 2000, 2001, 2001, 2002, 2002],
        "Season": ["Kharif     ", "Kharif", "Kharif", "Kharif", "Rabi", "Rabi", "Rabi"],
        "Crop": ["Rice", "Rice", "Rice", "Rice", "Wheat", "Wheat", "Wheat"],
        "Area": [10.0, 10.0, 10.0, 0.0, 20.0, 20.0, 5.0],
        "Production": [30.0, 30.0, np.nan, 5.0, 40.0, 0.0, 10.0],
    })


class TestCleaning(unittest.TestCase):
    def test_cleaning_rules(self):
        clean, report = dp.clean_data(dp.standardise_columns(make_raw_sample()))
        # Rows 0 and 1 become duplicates after trimming the season text -> 1 kept.
        # Row 2 has missing production, row 3 has zero area, row 5 has zero production.
        self.assertEqual(len(clean), 3)
        self.assertEqual(report["dropped_duplicates"], 1)
        self.assertEqual(report["dropped_missing_values"], 1)
        self.assertEqual(report["dropped_zero_or_negative_area"], 1)
        self.assertEqual(report["dropped_invalid_production"], 1)
        self.assertTrue((clean["Season"] == clean["Season"].str.strip()).all())

    def test_yield_formula(self):
        clean, _ = dp.clean_data(dp.standardise_columns(make_raw_sample()))
        np.testing.assert_allclose(clean["Yield"], clean["Production"] / clean["Area"])

    def test_production_not_a_model_input(self):
        self.assertNotIn("Production", dp.FEATURE_COLUMNS)
        self.assertNotIn("Yield", dp.FEATURE_COLUMNS)

    def test_missing_column_error(self):
        with self.assertRaises(ValueError):
            dp.standardise_columns(pd.DataFrame({"foo": [1]}))


class TestSplit(unittest.TestCase):
    def test_split_is_chronological(self):
        df = pd.DataFrame({"Year": np.repeat(np.arange(2000, 2010), 5)})
        train, test, info = tm.split_by_year(df)
        self.assertLess(train["Year"].max(), test["Year"].min())
        self.assertEqual(len(train) + len(test), len(df))
        self.assertEqual(info["test_years"], [2008, 2009])


class TestRecommendation(unittest.TestCase):
    def test_performance_index(self):
        df = pd.DataFrame({
            "State": ["A"] * 6 + ["B"] * 6,
            "District": ["D1"] * 12,
            "Crop": ["Rice"] * 12,
            "Season": ["Kharif"] * 12,
            "Year": list(range(2000, 2006)) * 2,
            "Area": [10.0] * 12,
            "Production": [40.0] * 6 + [20.0] * 6,
        })
        df["Yield"] = df["Production"] / df["Area"]
        out = rec.recommend_crops(df, "A", min_records=3)
        self.assertEqual(len(out), 1)
        self.assertGreater(out.loc[0, "Performance_Index"], 1.0)  # A is above the all-India median


class TestDatabase(unittest.TestCase):
    def test_save_and_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "test.db")
            db.save_prediction({"state": "A", "district": "D", "crop": "Rice", "season": "Kharif",
                                "year": 2020, "area": 10.0, "predicted_yield": 3.0,
                                "estimated_production": 30.0, "model_name": "x"}, path)
            history = db.fetch_history(db_path=path)
            self.assertEqual(len(history), 1)
            db.clear_history(path)
            self.assertTrue(db.fetch_history(db_path=path).empty)


@unittest.skipUnless(HAS_MODEL, "Train the model first: python data_preprocessing.py && python train_model.py")
class TestTrainedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = dp.load_clean_data()
        cls.model = pred.load_model()
        row = cls.df.iloc[-1]
        cls.row = row

    def _predict(self, **overrides):
        args = dict(state=self.row["State"], district=self.row["District"], crop=self.row["Crop"],
                    season=self.row["Season"], year=int(self.row["Year"]), area=100.0)
        args.update(overrides)
        return pred.predict_yield(self.model, **args)

    def test_prediction_is_valid_number(self):
        result = self._predict()
        self.assertTrue(np.isfinite(result["yield_per_unit_area"]))
        self.assertGreaterEqual(result["yield_per_unit_area"], 0)

    def test_total_production_is_yield_times_area(self):
        result = self._predict(area=250.0)
        self.assertAlmostEqual(result["estimated_production"], result["yield_per_unit_area"] * 250.0, places=6)

    def test_invalid_area_rejected(self):
        for bad_area in (0, -5, float("nan")):
            with self.assertRaises(ValueError):
                self._predict(area=bad_area)

    def test_unseen_district_does_not_crash(self):
        result = self._predict(district="A District That Does Not Exist")
        self.assertTrue(np.isfinite(result["yield_per_unit_area"]))

    def test_crop_details_and_explanation(self):
        details = pred.get_crop_details(self.df, self.row["Crop"], self.row["State"], self.row["District"])
        self.assertGreater(details["national"]["records"], 0)
        text = pred.explain_prediction(self._predict(), details, self.row["State"], self.row["District"],
                                       self.row["Season"], int(self.row["Year"]), 100.0, True,
                                       int(self.df["Year"].max()))
        self.assertIn("not a guaranteed harvest", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
