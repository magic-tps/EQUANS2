# EQUANS2 — Volt Patrol

Volt Patrol ordena suministros para priorizar inspecciones eléctricas. El pipeline audita siete Excel locales, construye variables de consumo anteriores a cada intervención, entrena modelos PU de CatBoost y LightGBM, valida con eventos de otro año y genera un ranking para `ALIMENTADOR_2025.xlsx`. El puntaje expresa prioridad relativa; no es una probabilidad calibrada ni confirma una vulneración.

## Ejecución local

Usa Python 3.11–3.14. Coloca exclusivamente estos archivos originales en `data/`:

`ALIMENTADOR_2024.xlsx`, `ALIMENTADOR_2025.xlsx`, `BALANCE_SED.xlsx`, `CALIDAD_TENSION.xlsx`, `FALLAS_SED.xlsx`, `HISTORICO_CNR.xlsx`, `POTENCIA_SED.xlsx`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
python -m src.benchmark
python -m streamlit run app.py
```

`main.py` produce el bundle local `models/final_model.joblib`, el ranking completo y TOP 76 en `outputs/`, y las métricas en `reports/`. La aplicación usa ese bundle para evaluar archivos nuevos sin entrenar de nuevo. Cada archivo cargado se procesa en memoria de la sesión.

## Interpretación y límites

- Las intervenciones confirmadas son positivas observadas. Los otros suministros están sin etiqueta; no se consideran negativos verificados.
- La validación usa positivos 2024 para entrenamiento y positivos 2025 para evaluación, con población sin etiqueta de cada año. El histórico no comparte suministros con el alimentador objetivo. Hits@K y Average Precision describen recuperación observable en esa comparación, no desempeño real confirmado en el alimentador objetivo.
- La fecha de corte precede cada intervención. Sólo se emplean lecturas anteriores al corte. Como los metadatos técnicos del histórico no tienen una fecha de observación propia, el modelo predictivo usa únicamente variables de consumo. Los datos SED se preparan para contexto descriptivo, con joins temporales; no definen etiquetas individuales.
- Los meses con consumo faltante, días inválidos o fecha inválida permanecen faltantes. Se requiere un mínimo de tres lecturas válidas para emitir score.
- El ranking 2025, el modelo y los reportes son artefactos locales derivados de datos confidenciales. `.gitignore` excluye Excel, CSV, joblib, reportes y secretos. Este repositorio público contiene código y documentación sin resultados individuales. Para desplegar Streamlit con inferencia real, el operador debe proporcionar el bundle entrenado y los datos autorizados por un canal privado compatible con las reglas del concurso. Véase [DEPLOYMENT.md](DEPLOYMENT.md).

## Estructura

- `src/`: auditoría, etiquetas, transformación mensual, variables, validación, entrenamiento e inferencia.
- `dashboard/` y `app.py`: interfaz Streamlit, validador y carga de Excel/CSV.
- `reports/`: auditoría y métricas agregadas; archivos individuales y catálogos quedan fuera de Git.
- `tests/`: pruebas de temporalidad, valores inválidos, validación y paridad de inferencia.

## Reentrenamiento

Ejecutar `python main.py` vuelve a auditar los siete Excel y entrena desde cero. El código no carga modelos, rankings ni métricas anteriores para entrenar. Las semillas y configuración elegida se guardan en el bundle. El diseño valida la arquitectura antes de ajustar el modelo final con todos los eventos elegibles.
