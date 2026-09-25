from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from dashboard.inference import infer,to_csv,to_xlsx
from dashboard.validators import validate
from src.cohorts import assert_disjoint,supply_partition
from src.features import consumption_features
from src.preprocess import monthly_long
from test_pipeline import sample

ROOT=Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=joblib.load(ROOT/"runtime/models/final_model.joblib")

    def test_runtime_manifest_and_sources(self):
        manifest=json.loads((ROOT/"runtime/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["trained_at"],self.bundle["trained_at"])
        actual={p.relative_to(ROOT/"runtime").as_posix() for p in (ROOT/"runtime").rglob("*") if p.is_file() and p.name!="manifest.json"}
        self.assertEqual(actual,set(manifest["files"]))
        for name,record in manifest["files"].items():
            content=(ROOT/"runtime"/name).read_bytes()
            self.assertEqual(len(content),record["bytes"],name)
            self.assertEqual(hashlib.sha256(content).hexdigest(),record["sha256"],name)
        for name,digest in manifest["source_sha256"].items():
            path=ROOT/"data"/name
            if path.exists():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)

    def test_calendar_gaps_do_not_form_adjacent_drops(self):
        raw=sample().iloc[:1].copy()
        raw.loc[0,"CONSUMO FEBRERO"]=np.nan
        raw["cutoff"]=pd.Timestamp("2025-04-01")
        features=consumption_features(raw,monthly_long(raw,2025))
        self.assertEqual(features.loc[0,"w3_available"],2)
        self.assertTrue(pd.isna(features.loc[0,"w3_drop"]))
        self.assertEqual(features.loc[0,"w3_decrease_streak"],0)

    def test_intervention_month_excluded_even_if_reading_is_earlier(self):
        raw=sample().iloc[:1].copy()
        raw["cutoff"]=pd.Timestamp("2025-03-25")
        features=consumption_features(raw,monthly_long(raw,2025))
        self.assertEqual(features.loc[0,"months_observed"],2)
        self.assertEqual(features.loc[0,"w3_last"],2)

    def test_partial_year_without_sed_and_stale_readings(self):
        raw=sample().drop(columns="SED_ID")
        result=infer(self.bundle,raw,2025).set_index("SUMINISTRO_ID")
        self.assertTrue(result.loc["A","valid_prediction"])
        self.assertFalse(result.loc["B","valid_prediction"])
        self.assertTrue(pd.isna(result.loc["B","priority_score"]))
        stale=infer(self.bundle,raw,2025,cutoff="2026-01-01")
        self.assertEqual(len(stale),2)
        self.assertFalse(stale.valid_prediction.any())
        self.assertTrue(stale.priority_score.isna().all())
        self.assertTrue(stale["rank"].isna().all())

    def test_trimmed_duplicate_ids_are_rejected(self):
        raw=sample();raw["SUMINISTRO_ID"]=["A"," A "]
        self.assertTrue(validate(raw)[0])
        with self.assertRaises(ValueError):infer(self.bundle,raw,2025)

    def test_method_abstains_and_features_exclude_outcomes(self):
        gate=self.bundle["method_classifier"]["summary"]
        if not gate["enabled"]:
            result=infer(self.bundle,sample(),2025)
            self.assertTrue(result.method_hypothesis.eq("No determinable con estos datos").all())
        for col in self.bundle["feature_columns"]:
            self.assertTrue(col.startswith(("w3_","w6_","w12_","months_")),col)

    def test_partition_guards_reject_overlap(self):
        rows=pd.DataFrame({"SUMINISTRO_ID":["A","A"],"split":["train","test"]})
        with self.assertRaises(ValueError):assert_disjoint(rows)
        rows.loc[1,"SUMINISTRO_ID"]="B"
        assert_disjoint(rows)
        self.assertEqual(supply_partition("stable"),supply_partition("stable"))
        path=ROOT/"reports/private/cohort_features.csv"
        if path.exists():assert_disjoint(pd.read_csv(path))

    def test_csv_and_excel_neutralize_formulas(self):
        raw=pd.DataFrame({"notes":["=1+1"," +SUM(A1:A2)","@SUM(A1)","normal"],"value":[-2,3,4,5]})
        csv=pd.read_csv(BytesIO(to_csv(raw)))
        excel=pd.read_excel(BytesIO(to_xlsx(raw)))
        for result in (csv,excel):
            self.assertTrue(result.loc[0,"notes"].startswith("'="))
            self.assertTrue(result.loc[1,"notes"].startswith("' +"))
            self.assertEqual(result.loc[3,"notes"],"normal")
            self.assertEqual(result.loc[0,"value"],-2)

    def test_published_ranking_matches_real_inference(self):
        raw=pd.read_csv(ROOT/"runtime/data/ALIMENTADOR_2025.csv")
        # Include high/low scores and supplies with invalid recent months.
        official=pd.read_csv(ROOT/"runtime/outputs/VOLT_PATROL_RANKING_COMPLETO.csv").set_index("SUMINISTRO_ID")
        ids=list(official.head(8).index)+list(official.tail(8).index)
        result=infer(self.bundle,raw[raw.SUMINISTRO_ID.isin(ids)],2025,cutoff="2026-01-01").set_index("SUMINISTRO_ID")
        np.testing.assert_allclose(result.priority_score,official.loc[result.index,"priority_score"],equal_nan=True,atol=1e-12)
        self.assertEqual(set(result.index),set(ids))
        self.assertEqual(result.valid_prediction.tolist(),official.loc[result.index,"valid_prediction"].tolist())


if __name__=="__main__":unittest.main()
