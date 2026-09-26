# Rúbrica: evidencia implementada y límites verificables

## A1 · Aciertos en los primeros 76 — 30%

La entrega es una lista de identificadores ordenados, con Top 76 consistente y sin duplicados. Se reúnen los suministros 2024/2025 usando la información más reciente de cada uno; se conservan alternativas anuales porque no se recibió una plantilla adicional. Los 76 hurtos informados por la lámina pertenecen al conjunto ciego: no son 76 etiquetas disponibles ni un resultado de nuestro modelo.

La pantalla de entrega acepta resultados verificados y calcula precisión @76 sólo con las 76 etiquetas. Los aciertos históricos P/U son recuperación observable, no precisión oficial. **El resultado A1 seguirá pendiente hasta la evaluación del concurso.**

## B1 · Solidez técnica — 15%

Pipeline temporal, meses faltantes conservados, identificadores separados entre particiones, preprocesador ajustado en entrenamiento y congelado para inferencia. Pruebas de paridad, serialización, CSV/Excel, datos inválidos y cobertura completa. Los empates de evaluación usan hash de ID sin etiquetas, evitando que una regla con valores iguales se beneficie de que los positivos estén primero en el archivo.

La versión vectorizada reproduce el cálculo de referencia, incluidos signos numéricos cercanos a cero relevantes para árboles existentes. Se comprueban scores sobre toda la población objetivo. SHAP corresponde al componente con mayor peso del modelo, informado explícitamente; no se muestran explicaciones de un componente de peso cero como si decidiera la prioridad.

## B2 · Encaje operativo — 20%

Planificación mensual con capacidad y cupos exploratorios, selección conservada con versión del modelo, responsable, estado, fecha, evidencia, hallazgo, costos y recuperos. Las campañas se guardan como una cartera JSON y se restauran en otra sesión. Una confirmación exige fecha y referencia de evidencia. Las anotaciones no se convierten automáticamente en etiquetas de entrenamiento.

Se exportan ranking, fichas y planes. La cartera es un archivo bajo control del usuario; hay que guardarlo antes de salir. No se afirma que el disco de Streamlit Cloud conserve las sesiones permanentemente.

## B3 · Escalabilidad — 15%

**Capacidad medida:** procesamiento de un millón de suministros mediante CSV por lotes, transformaciones vectorizadas y ordenamiento global en disco. Verificación completa de filas y orden, detección de duplicados entre lotes, salida atómica y limpieza de temporales. El reporte registra tiempo, pico de RSS muestreado, tamaño de datos y equipo. Los datos de carga son réplicas sintéticas y no prueban precisión. La medición corresponde al equipo local, no a una garantía de RAM en Cloud.

**Zonas:** tres particiones por SED con cero SED compartidas entre entrenamiento y evaluación, junto con dos periodos posteriores. Configuración fija para auditoría; no es una nueva prueba ciega ni una selección a posteriori del mejor modelo. La población sin etiqueta disponible pertenece a un único alimentador; para demostrar transferencia entre alimentadores completos hacen falta datos e inspecciones de otras zonas.

**Sin historial:** entrada aceptada, conservación de IDs, estado explícito y visitas exploratorias con rotación mensual por SED. No se atribuye una probabilidad de hurto a casos sin evidencia temporal. La prueba enmascara 0, 1, 2 y 3 meses y comprueba el comportamiento. La cobertura operativa está implementada; la precisión predictiva sin historial no puede demostrarse con estos datos.

## B4 · Sostenibilidad — 10%

Dependencias fijadas, modelo versionado, manifiesto por archivo, pruebas en Linux y Windows y comandos para repetir entrenamiento, auditoría y publicación. Monitoreo de cobertura y PSI con referencia fija; umbral de revisión 0,20 documentado como regla operativa, no prueba de deterioro ni disparador automático.

Registro de costos y recuperos reales por visita. Sólo se calcula beneficio observado cuando los datos están completos. Simulador económico sin valores prellenados: los supuestos provienen del operador y se distinguen de resultados. Reentrenamiento requiere resultados verificados, periodo reservado y comparación con la versión vigente; nunca se ejecuta al cargar un archivo del jurado.

## B5 · Innovación — 10%

Prioridad explicada, abstención cuando la evidencia es insuficiente, campañas trazables y medición de utilidad. Comparación reproducible con reglas simples y selección aleatoria; no se atribuyen esas reglas al proceso real de EQUANS. La aplicación permite aportar la lista actual y etiquetas verificadas para comparar aciertos Top 76 sobre la misma población.

La valoración de innovación corresponde al jurado. Ni complejidad ni aspecto visual sustituyen evidencia de beneficio.

## Evidencia publicada

- [Entrega y alcance](runtime/reports/delivery_manifest.json).
- [Prueba de un millón](runtime/reports/scale_benchmark.json).
- [SED retenidas](runtime/reports/geographic_validation.csv) y [periodos posteriores](runtime/reports/rolling_validation.csv).
- [Reglas de referencia](runtime/reports/operational_baselines.csv).
- [Cobertura sin historial](runtime/reports/cold_start_validation.json).
- [Límites de las auditorías](runtime/reports/robustness_summary.json).

Los puntos del jurado y la precisión del conjunto ciego no pueden garantizarse antes de su evaluación.
