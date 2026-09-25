# EQUANS2 — Volt Patrol

Volt Patrol prioriza inspecciones eléctricas con evidencia de consumo. Incluye un dashboard operativo, un ensamble PU de CatBoost y LightGBM y un experimento de identificación de métodos con abstención cuando la evidencia no alcanza. El índice expresa prioridad relativa; no es una probabilidad calibrada ni confirma una vulneración.

**[Abrir el dashboard](https://magic-tps-equans2-app-cyow2q.streamlit.app/)**

## Dashboard 2.0

- **Centro de control:** cobertura, primera ronda, señales observadas, concentración por SED y mapa de consumo.
- **Priorizar inspecciones:** búsqueda, filtros, selección de suministros, estados, notas y exportación del plan.
- **Investigar suministro:** curvas 2024/2025, comparación con pares, calidad de lecturas, aportes SHAP, variación entre semillas y ficha de campo imprimible.
- **Métodos observados:** seis familias documentadas, prueba del clasificador, matriz de confusión y criterios de abstención.
- **Red y pérdidas:** balance de energía por SED, potencia, fallas y calidad de tensión.
- **Laboratorio de modelos:** selección, prueba final, control comparable, estabilidad y diferencias de población.
- **Evaluar nueva data:** Excel/CSV, corte configurable, validación, inferencia y descargas sin reentrenamiento.
- **Calidad y trazabilidad:** lecturas faltantes, diccionario, cobertura y comprobación de particiones.

## Ejecución local

Usa Python 3.12–3.14; el paquete se generó y verificó con Python 3.14. El repositorio incluye en `runtime/` los CSV, el modelo y los reportes necesarios para abrir todas las pantallas. No necesitas los Excel originales para ejecutar la aplicación.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

Abre **http://localhost:8501**. La aplicación usa los artefactos locales si existen y, en una instalación limpia, los de `runtime/`. Cada archivo cargado y las notas del plan se mantienen en la sesión; descarga el resultado para conservarlo.

## Modelo y resultados comprobados

Se compararon tres conjuntos de variables, dos algoritmos y mezclas del ensamble. La arquitectura se eligió por Hits@76 y luego AP en selección, antes de consultar la prueba final. Cada componente final promedia cinco semillas con remuestreo de la población sin etiqueta.

| Prueba final | Modelo elegido: patrón relativo de 3 meses | Control multiventana |
|---|---:|---:|
| Positivos conocidos en Top 76 | 75 | 74 |
| Recall conocido @76 | 3,84% | 3,79% |
| AP observable | 0,822 | 0,861 |

El modelo elegido recuperó un caso conocido adicional en Top 76, pero tuvo menor AP. Esto no demuestra una mejora general ni confirma infractores en la población objetivo.

El ranking conserva **4.904 suministros**: **4.561** con score y **343** pendientes de datos válidos recientes. La coincidencia media del Top 76 entre semillas es **73,0%**.

El clasificador de método obtuvo **F1 macro 0,216** y **23,9% de exactitud**, frente a **57,4%** de la referencia mayoritaria. Por ello, la atribución individual está **deshabilitada**: se muestra «No determinable con estos datos». El catálogo histórico sí conserva las familias documentadas. [Reporte completo](runtime/reports/final_report.md).

## Interpretación y límites

- Las intervenciones confirmadas son positivas observadas. Los otros suministros están sin etiqueta; no se consideran negativos verificados.
- La validación separa suministros: positivos de 2024 para entrenamiento, enero–junio de 2025 para selección y julio–diciembre de 2025 para prueba. Los suministros sin etiqueta se separan por identificador y sus cortes se emparejan por periodo. No hay identificadores compartidos entre particiones. Hits@K y AP miden recuperación observable, no precisión real de hurto.
- La fecha de corte precede cada intervención. Sólo se emplean lecturas anteriores al corte. Como los metadatos técnicos del histórico no tienen una fecha de observación propia, el modelo predictivo usa únicamente variables de consumo. Los datos SED se preparan para contexto descriptivo, con joins temporales; no definen etiquetas individuales.
- Los meses con consumo faltante, días inválidos o fecha inválida permanecen faltantes. Se requieren los tres meses de calendario completos anteriores al corte; se excluyen el mes de intervención y las lecturas futuras. Los huecos no se convierten en meses consecutivos.
- El AUC del clasificador de origen es 0,756. Mezcla diferencias de origen y de etiqueta; no es una medida pura de transferencia. Se necesitan resultados de inspección del alimentador objetivo para validar el rendimiento operativo.
- Por autorización del responsable del proyecto, `runtime/` publica los consumos con identificadores de suministro, contexto de red, ranking, explicaciones, modelo y reportes necesarios para Streamlit. Los Excel originales, eventos históricos individuales, cohortes de entrenamiento, cargas de usuarios y secretos quedan fuera de Git. Véase [DEPLOYMENT.md](DEPLOYMENT.md).

## Estructura

- `src/`: auditoría, etiquetas, transformación mensual, variables, validación, entrenamiento e inferencia.
- `dashboard/` y `app.py`: interfaz Streamlit, validador y carga de Excel/CSV.
- `runtime/`: paquete público de ejecución, con manifiesto SHA-256 por archivo.
- `data/`, `outputs/`, `models/`, `reports/`: originales y resultados locales de entrenamiento; excluidos de Git.
- `tests/`: pruebas de temporalidad, valores inválidos, validación y paridad de inferencia.

## Reentrenamiento

Coloca en `data/` los siete originales: `ALIMENTADOR_2024.xlsx`, `ALIMENTADOR_2025.xlsx`, `BALANCE_SED.xlsx`, `CALIDAD_TENSION.xlsx`, `FALLAS_SED.xlsx`, `HISTORICO_CNR.xlsx` y `POTENCIA_SED.xlsx`.

```powershell
pip install -r requirements-train.txt
python main.py
python -m src.package_runtime
python -m unittest discover -s tests -v
python -m src.benchmark
```

`main.py` audita los originales y entrena desde cero, sin cargar modelos, rankings ni métricas anteriores. `src.package_runtime` actualiza únicamente la lista explícita de archivos publicables, verifica cobertura y particiones y genera el manifiesto. Conserva juntos modelo, reportes y dependencias de la misma ejecución.
