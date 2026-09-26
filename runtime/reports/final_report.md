# Volt Patrol 2 — resultados reproducibles

La validación anterior reutilizaba suministros U entre años. Esta ejecución corrige el solapamiento: ningún ID cruza entrenamiento, selección y prueba. Positivos: 2024 para entrenar, enero–junio 2025 para elegir y julio–diciembre 2025 para la prueba final. U se divide por hash de ID y sus cortes se emparejan por periodo. El mes de intervención se excluye y los meses faltantes conservan su posición. Se exigen tres meses completos recientes.

## Modelo elegido

Variables: `patron_relativo_3_meses`. Selección por Hits@76 y luego AP, antes de consultar la prueba final.

| Métrica de prueba | Elegido | Control multiventana |
|---|---:|---:|
| Hits@76 | 75 | 75 |
| Recall conocido@76 | 3.84% | 3.84% |
| AP observable | 0.822 | 0.864 |

AUC ROC observable del detector: 0.6804863086213853. Se calcula con positivos históricos frente a población sin etiqueta y no debe confundirse con el AUC del clasificador de origen.

El control se reentrenó con la misma partición. Los empates de evaluación se resuelven por hash de ID, sin usar etiquetas ni el orden original. Una mejora puntual en Top 76 no demuestra superioridad general.

Bootstrap por SED: 200 repeticiones; intervalo 95% de Hits@76 72.0–76.0. Describe variación dentro de esta cohorte, no incertidumbre de transferencia a otra población.

AUC de separabilidad de origen: 0.756. Este diagnóstico mezcla dominio y etiqueta. La prueba histórica no demuestra detección real en el alimentador objetivo. U no son negativos verificados; Hits, AP y lift son observables. La dispersión entre semillas no es un intervalo de probabilidad de hurto.

## Identificación del método

F1 macro: 0.214. Exactitud: 23.7%; referencia mayoritaria: 57.4%. Casos de prueba: 1952. Atribución individual habilitada: no. La atribución de métodos se bloquea si no supera los criterios definidos antes de evaluar. Sólo una inspección puede confirmar una vulneración.

## Cobertura y explicación

Ranking: 4904 suministros, 4561 scores válidos. Coincidencia media del Top 76 entre semillas: 71.7%. Las explicaciones se regeneran con `src.delivery` para el componente con mayor peso; SHAP no implica causalidad. El paquete autorizado `runtime/` incluye los CSV, modelo y reportes necesarios para Streamlit. Los originales y las cohortes con intervenciones individuales permanecen locales.
