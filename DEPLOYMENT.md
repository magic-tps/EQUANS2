# Despliegue de Volt Patrol

Este repositorio público contiene código y configuración de Streamlit. No contiene los Excel, el ranking ni `final_model.joblib` porque se derivan de información de suministros e intervenciones.

La aplicación pública abre y muestra instrucciones si falta el bundle. La inferencia real requiere proporcionar el bundle entrenado por un canal autorizado. Una opción es desplegar desde una copia privada de este repositorio en Streamlit Community Cloud, con acceso limitado al jurado autorizado y con los archivos permitidos en esa copia privada. Antes de usar esa opción, el responsable del concurso debe confirmar que permite distribuir el bundle y los datos derivados a ese grupo.

Para un despliegue privado:

1. Ejecutar `python main.py` localmente con los siete Excel autorizados.
2. Copiar `models/final_model.joblib` y los archivos locales necesarios a la copia privada. Mantenerlos fuera del repositorio público.
3. Configurar `app.py` como archivo principal y `requirements.txt` como dependencias.
4. Ejecutar `python -m unittest discover -s tests -v` y probar un Excel/CSV de entrada antes de compartir el enlace privado.

El bundle serializado debe cargarse sólo desde una fuente confiable. La interfaz no guarda de forma permanente los archivos que sube el usuario ni los comparte mediante caché global de datos.
