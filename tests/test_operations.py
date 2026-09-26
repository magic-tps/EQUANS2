from __future__ import annotations
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd

from dashboard.inference import infer,read_upload,to_xlsx
from src.batch import rank_batches
from src.campaigns import empty_portfolio,create_campaign,update_entries,dumps,loads,campaign_metrics,exploration_queue
from src.features import consumption_features,_consumption_features_reference
from src.preprocess import monthly_long
from src.monitoring import evaluate_top,drift,scenario
from src.validation import observable_metrics

ROOT=Path(__file__).resolve().parents[1]


class OperationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=joblib.load(ROOT/"runtime/models/final_model.joblib")
        cls.raw=pd.read_csv(ROOT/"runtime/data/ALIMENTADOR_2025.csv")
        cls.ranking=pd.read_csv(ROOT/"runtime/outputs/VOLT_PATROL_RANKING_COMPLETO.csv")

    def test_vectorized_features_match_reference_with_gaps_and_variable_cuts(self):
        raw=self.raw.head(50).copy()
        raw["cutoff"]=[pd.Timestamp(2025,5+i%8,15) for i in range(len(raw))]
        raw.loc[0,"CONSUMO ABRIL"]=np.nan
        raw.loc[1,"CONSUMO MARZO"]=0
        long=monthly_long(raw,2025)
        a=consumption_features(raw,long);b=_consumption_features_reference(raw,long)
        cols=b.select_dtypes("number").columns
        np.testing.assert_allclose(a[cols],b[cols],rtol=1e-8,atol=1e-8,equal_nan=True)

    def test_entire_target_scores_remain_compatible(self):
        result=infer(self.bundle,self.raw,2025,cutoff="2026-01-01").set_index("SUMINISTRO_ID")
        old=self.ranking.set_index("SUMINISTRO_ID").loc[result.index]
        np.testing.assert_allclose(result.priority_score,old.priority_score,atol=1e-10,equal_nan=True)
        self.assertEqual(result.valid_prediction.tolist(),old.valid_prediction.tolist())

    def test_batch_order_is_global_and_independent_of_batch_size(self):
        raw=self.raw.head(40)
        expected=infer(self.bundle,raw,2025,cutoff="2026-01-01")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"rank.csv"
            result=rank_batches(self.bundle,[raw.iloc[:17],raw.iloc[17:]],path,2025,"2026-01-01")
            actual=pd.read_csv(path)
            self.assertEqual(result["supplies"],40)
            self.assertEqual(actual.SUMINISTRO_ID.tolist(),expected.SUMINISTRO_ID.tolist())
            np.testing.assert_allclose(actual.priority_score,expected.priority_score,atol=1e-12,equal_nan=True)
            self.assertEqual(actual.position.tolist(),list(range(1,41)))
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_duplicate_across_batches_preserves_previous_output_and_cleans_up(self):
        raw=self.raw.head(3)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"rank.csv";path.write_text("previous",encoding="utf-8")
            with self.assertRaises(ValueError):rank_batches(self.bundle,[raw,raw],path,2025,"2026-01-01")
            self.assertEqual(path.read_text(),"previous")
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_no_history_is_preserved_and_not_scored(self):
        raw=self.raw.head(5)[["SUMINISTRO_ID","SED_ID"]]
        result=infer(self.bundle,raw,2025,cutoff="2026-01-01")
        self.assertEqual(set(result.SUMINISTRO_ID),set(raw.SUMINISTRO_ID))
        self.assertTrue(result.priority_score.isna().all())
        self.assertFalse(result.valid_prediction.any())
        self.assertTrue(result.review_action.str.contains("exploratoria").all())

    def test_campaign_roundtrip_results_and_exploration(self):
        portfolio=empty_portfolio()
        campaign=create_campaign(portfolio,self.ranking,"2026-10",10,2,self.bundle["version"],"test")
        self.assertEqual(len(campaign["entries"]),10)
        self.assertEqual(sum(e["purpose"].startswith("Exploración") for e in campaign["entries"]),2)
        frame=pd.DataFrame(campaign["entries"])
        frame.loc[0,"outcome"]="Hurto confirmado"
        with self.assertRaises(ValueError):update_entries(campaign,frame)
        self.assertEqual(campaign["entries"][0]["outcome"],"Pendiente")
        frame.loc[0,"evidence"]="ACTA-001";frame.loc[0,"visit_date"]="2026-10-05"
        frame.loc[0,"status"]="Visitado";frame.loc[0,"inspection_cost"]=50.;frame.loc[0,"recovered_amount"]=300.
        update_entries(campaign,frame)
        restored=loads(dumps(portfolio))
        self.assertEqual(restored,portfolio)
        metrics=campaign_metrics(restored["campaigns"][0])
        self.assertEqual(metrics["confirmed"],1);self.assertEqual(metrics["resolution_coverage"],.1)
        self.assertEqual(metrics["net_observed"],250.)
        with self.assertRaises(ValueError):create_campaign(portfolio,self.ranking,"2026-10",10,2,"2.0.0","test")

    def test_exploration_is_reproducible_and_never_claims_risk(self):
        a=exploration_queue(self.ranking,"2026-10");b=exploration_queue(self.ranking,"2026-10")
        self.assertEqual(a.SUMINISTRO_ID.tolist(),b.SUMINISTRO_ID.tolist())
        self.assertTrue(a.priority_score.isna().all())
        self.assertEqual(a.head(a.SED_ID.nunique()).SED_ID.nunique(),a.SED_ID.nunique())

    def test_precision_requires_all_top76_labels(self):
        top=self.ranking.head(76)
        truth=pd.DataFrame({"SUMINISTRO_ID":top.SUMINISTRO_ID,"HURTO_REAL":[1]*38+[0]*38})
        self.assertIsNone(evaluate_top(self.ranking,truth.head(20))["precision_at_k"])
        self.assertEqual(evaluate_top(self.ranking,truth)["precision_at_k"],.5)
        truth.loc[truth.index[0],"HURTO_REAL"]=np.nan
        with self.assertRaises(ValueError):evaluate_top(self.ranking,truth)

    def test_monitoring_and_scenario_are_distinct_from_measurements(self):
        report=drift(self.ranking,self.ranking)
        np.testing.assert_allclose(report.psi,0,atol=1e-12)
        result=scenario(100,.1,20,500,100)
        self.assertEqual(result["expected_net"],2900)
        self.assertEqual(result["break_even_hit_rate"],.042)

    def test_delivery_covers_both_years_without_duplicate_ids(self):
        path=ROOT/"runtime/outputs/delivery/ENTREGA_RANKING.csv"
        if not path.exists():self.skipTest("Paquete de entrega aún no generado")
        delivered=pd.read_csv(path)
        year24=pd.read_csv(ROOT/"runtime/data/ALIMENTADOR_2024.csv")
        self.assertEqual(delivered.columns.tolist(),["SUMINISTRO_ID"])
        self.assertTrue(delivered.SUMINISTRO_ID.is_unique)
        self.assertEqual(set(delivered.SUMINISTRO_ID),set(year24.SUMINISTRO_ID)|set(self.raw.SUMINISTRO_ID))
        top=pd.read_csv(ROOT/"runtime/outputs/delivery/ENTREGA_TOP_76.csv")
        self.assertEqual(top.SUMINISTRO_ID.tolist(),delivered.head(76).SUMINISTRO_ID.tolist())

    def test_tied_scores_do_not_depend_on_label_grouped_input_order(self):
        ids=np.asarray([f"ID_{i}" for i in range(200)])
        labels=np.r_[np.ones(100),np.zeros(100)];scores=np.ones(200)
        first=observable_metrics(labels,scores,ids)
        reverse=observable_metrics(labels[::-1],scores[::-1],ids[::-1])
        self.assertEqual(first,reverse)
        self.assertLess(first["hits_76"],76)

    def test_published_curve_agrees_with_metrics_under_ties(self):
        curve=pd.read_csv(ROOT/"runtime/reports/recovery_curve.csv").set_index("position")
        for k in (10,25,50,76,100,200,500):
            self.assertEqual(int(curve.loc[k,"known_hits"]),self.bundle["validation"][f"hits_{k}"])

    def test_upload_preserves_identifier_leading_zeros(self):
        raw=pd.DataFrame({"SUMINISTRO_ID":["00123","00456"]})
        for name,content in (("ids.csv",raw.to_csv(index=False).encode()),("ids.xlsx",to_xlsx(raw))):
            self.assertEqual(read_upload(name,content).SUMINISTRO_ID.tolist(),raw.SUMINISTRO_ID.tolist())


if __name__=="__main__":unittest.main()
