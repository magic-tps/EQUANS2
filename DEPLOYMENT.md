# Despliegue de Volt Patrol

La aplicación publicada es [Volt Patrol](https://magic-tps-equans2-app-cyow2q.streamlit.app/), conectada al repositorio `magic-tps/EQUANS2`, rama `main`, entrada `app.py`.

## Paquete incluido

La publicación de los archivos necesarios fue autorizada por el responsable del proyecto. `runtime/` contiene seis CSV de consumos y red, ranking completo y Top 76, aportes SHAP, modelo y reportes. Incluye datos reales del proyecto con identificadores de suministro; no es una demostración con datos inventados.

`runtime/manifest.json` enumera los archivos, tamaños, hashes SHA-256 y versiones de dependencias. Los originales Excel y el histórico individual de intervenciones no son necesarios para consultar el dashboard ni para inferir sobre archivos nuevos, y no se incluyen. Las cohortes individuales de prueba se sustituyen por una curva agregada y una auditoría de particiones.

## Streamlit Community Cloud

1. Configura repositorio `magic-tps/EQUANS2`, rama `main` y archivo `app.py`.
2. Usa Python 3.12 o posterior; el entorno local verificado es Python 3.14. Las dependencias exactas están en `requirements.txt`.
3. Despliega. No hacen falta secretos ni conexiones externas para cargar datos o modelo.
4. Tras actualizar GitHub, verifica que la app muestre **VOLT PATROL 2.0**, modelo **2.0.0** y **4.904 suministros**. Si sigue mostrando una revisión antigua, revisa los logs o reinicia la app desde su panel de administración.

La versión de Python se selecciona en las opciones avanzadas de despliegue. Si necesitas cambiarla en una aplicación existente, sigue las [instrucciones oficiales de Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python). No dependemos de `runtime.txt` para cambiarla.

## Actualizar el paquete

```powershell
pip install -r requirements-train.txt
python main.py
python -m src.package_runtime
python -m unittest discover -s tests -v
```

El empaquetador comprueba que reportes y modelo correspondan al mismo entrenamiento, que el ranking preserve la población y que los identificadores no se repitan entre particiones. Revisa y publica juntos código y `runtime/`.

El bundle se carga desde una ruta fija del proyecto. La interfaz acepta Excel/CSV para inferencia; no acepta modelos serializados de usuarios. Los archivos cargados, resultados y notas permanecen en su sesión y no se comparten mediante la caché global. Descarga el plan o ranking antes de cerrar la sesión.
