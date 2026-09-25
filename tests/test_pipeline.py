from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dashboard.inference import infer,read_upload,to_xlsx
from dashboard.validators import validate
from src.features import build_features
from src.labels import classify_events
from src.predict import rank,score_features
from src.preprocess import before_cutoff,monthly_long
from src.utils import DATA_FILES
from src.artifacts import resolve

ROOT=Path(__file__).resolve().parents[1]


def sample():
    return pd.DataFrame({"SUMINISTRO_ID":["A","B"],"SED_ID":["S","S"],
                         "CONSUMO ENERO":[0,100],"DIAS FACTURADO\nENERO":[30,0],
                         "FECHA LECTURA ENERO":["2025-01-20","2025-01-20"],
                         "CONSUMO FEBRERO":[60,80],"DIAS FACTURADO\nFEBRERO":[30,30],
                         "FECHA LECTURA FEBRERO":["2025-02-20","2025-02-20"],
                         "CONSUMO MARZO":[90,70],"DIAS FACTURADO\nMARZO":[30,30],
                         "FECHA LECTURA MARZO":["2025-03-20","2025-03-20"]})


class PipelineTests(unittest.TestCase):
    def test_source_workbooks_open_and_unchanged(self):
        paths=[ROOT/"data"/name for name in DATA_FILES]
        if not all(p.exists() for p in paths):self.skipTest("Excel locales no disponibles")
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for p in paths:
            with pd.ExcelFile(p) as book:
                self.assertGreater(len(book.sheet_names),0)
        after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        self.assertEqual(before,after)

    def test_zero_is_distinct_from_invalid_days(self):
        long=monthly_long(sample(),2025)
        a=long[(long.SUMINISTRO_ID=="A") & (long.nominal_month.dt.month==1)].iloc[0]
        b=long[(long.SUMINISTRO_ID=="B") & (long.nominal_month.dt.month==1)].iloc[0]
        self.assertEqual(a.daily_kwh,0)
        self.assertEqual(a.observation_status,"observed_zero")
        self.assertTrue(pd.isna(b.daily_kwh))
        self.assertEqual(b.observation_status,"invalid")

    def test_future_reading_is_excluded(self):
        long=monthly_long(sample(),2025)
        cut=before_cutoff(long,pd.Timestamp("2025-03-01"))
        self.assertFalse(cut.nominal_month.dt.month.eq(3).any())
        base=sample()
        base["cutoff"]=pd.Timestamp("2025-03-01")
        f=build_features(base,long)
        self.assertEqual(f.loc[0,"months_observed"],2)
        self.assertEqual(f.loc[1,"months_observed"],1)

    def test_inconsistent_reading_is_missing_not_zero(self):
        frame=sample()
        frame.loc[0,"FECHA LECTURA ENERO"]="2024-09-01"
        long=monthly_long(frame,2025)
        row=long[(long.SUMINISTRO_ID=="A") & (long.nominal_month.dt.month==1)].iloc[0]
        self.assertEqual(row.observation_status,"inconsistent_reading")
        self.assertTrue(pd.isna(row.daily_kwh))

    def test_peer_reference_excludes_self(self):
        frame=sample()
        frame["cutoff"]=pd.Timestamp("2025-04-01")
        feature=build_features(frame,monthly_long(frame,2025))
        self.assertAlmostEqual(feature.loc[0,"peer_difference"],
                               feature.loc[0,"w6_median"]-feature.loc[1,"w6_median"])

    def test_sed_join_respects_cutoff(self):
        frame=sample().iloc[:1].copy()
        frame["cutoff"]=pd.Timestamp("2025-06-01")
        balance=pd.DataFrame({"SED_ID":["S","S"],"PERIODO BALANCE":["2025-05-01","2025-07-01"],
                              "E. DISTRIBUIDA MES MWh":[10,999],"E. FACTURADA MES MWh":[8,999],
                              "PERDIDA  MES MWh":[2,0],"PERDIDA MES %":[.2,0],
                              "PROYEC_PERDIDAS NO TECNICAS MES MWh":[1,0],
                              "CANT CLIENTES TOTALES APROX":[2,2]})
        feature=build_features(frame,monthly_long(frame,2025),{"BALANCE_SED":balance})
        self.assertEqual(feature.loc[0,"sed_distributed_mwh"],10)

    def test_labels_require_explicit_evidence(self):
        h=pd.DataFrame({"SUMINISTRO_ID":["A","B","C"],"DIA INTERVENCION":["2025-05-01"]*3,
                        "VULERACION ENCONTRADA":["CONEX DIRECTA 2 LINEAS","SIN HALLAZGO","RETIRO"],
                        "TIPIFICACION":["MANIPULACION","MANIPULACION","SUMINISTROS RETIRADOS"]})
        labels=classify_events(h).label_class.tolist()
        self.assertEqual(labels,["POSITIVE_CONFIRMED","AMBIGUOUS","INVALID"])

    def test_validator_and_upload(self):
        frame=sample()
        errors,info=validate(frame)
        self.assertFalse(errors)
        self.assertEqual(info["supplies"],2)
        csv=frame.to_csv(index=False).encode()
        self.assertEqual(len(read_upload("input.csv",csv)),2)
        self.assertEqual(len(read_upload("input.xlsx",to_xlsx(frame))),2)
        self.assertTrue(validate(frame.drop(columns=["CONSUMO MARZO"]))[0])
        self.assertTrue(validate(pd.concat([frame,frame.iloc[:1]]))[0])
        with self.assertRaises(ValueError):read_upload("empty.csv",b"")

    def test_bundle_roundtrip_and_inference(self):
        path=resolve(ROOT,"models/final_model.joblib")
        if not path.exists():self.skipTest("Modelo local aún no entrenado")
        bundle=joblib.load(path)
        source=resolve(ROOT,"data/ALIMENTADOR_2025.xlsx")
        frame=(pd.read_excel(source) if source.suffix==".xlsx" else pd.read_csv(source)).head(5)
        a=infer(bundle,frame,2025)
        with tempfile.TemporaryDirectory() as directory:
            copy=Path(directory)/"bundle.joblib"
            joblib.dump(bundle,copy)
            b=infer(joblib.load(copy),frame,2025)
        np.testing.assert_allclose(a.priority_score,b.priority_score,equal_nan=True)
        self.assertEqual(a.SUMINISTRO_ID.nunique(),5)
        base=frame.copy()
        base["cutoff"]=pd.Timestamp("2026-01-01")
        features=build_features(base,monthly_long(base,2025))
        local=rank(base,score_features(bundle,features),features.months_observed,recent_months=features.w3_available)
        np.testing.assert_allclose(a.priority_score,local.priority_score,equal_nan=True)

    def test_ranking_covers_entire_target(self):
        path=resolve(ROOT,"outputs/VOLT_PATROL_RANKING_COMPLETO.csv")
        if not path.exists():self.skipTest("Ranking local aún no generado")
        source=resolve(ROOT,"data/ALIMENTADOR_2025.xlsx")
        target=(pd.read_excel(source) if source.suffix==".xlsx" else pd.read_csv(source))
        ranking=pd.read_csv(path)
        self.assertEqual(set(target.SUMINISTRO_ID),set(ranking.SUMINISTRO_ID))
        self.assertEqual(len(ranking),target.SUMINISTRO_ID.nunique())
        self.assertEqual(len(pd.read_csv(resolve(ROOT,"outputs/VOLT_PATROL_TOP_76.csv"))),76)

    def test_streamlit_pages_load(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Streamlit no instalado")
        for page_index in range(8):
            app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=40).run()
            if page_index:
                app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[page_index]).run()
            self.assertFalse(app.exception,f"Error en página {page_index}: {app.exception}")


if __name__=="__main__":unittest.main()
