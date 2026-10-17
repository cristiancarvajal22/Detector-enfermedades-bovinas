# Evaluación interna del modelo bovino

Esta ejecución prepara un nuevo modelo para el informe académico. No reemplaza automáticamente la exportación anterior utilizada por la aplicación.

## Procedimiento

1. `auditar.py` decodifica los cinco archivos de muestras e inventaría sus huellas. También compara las imágenes con las exportaciones locales de sapipise y kambingpise.
2. `preparar.py` agrupa copias próximas mediante pHash de 64 bits (distancia máxima 4) y familias de nombres Roboflow. Excluye grupos con etiquetas distintas e imágenes asociadas a kambingpise cuya diferencia media de píxeles es como máximo 8/255 tras ajustar el tamaño. Reserva 30 grupos por clase con semilla 42, utiliza una imagen de cada grupo para prueba y excluye sus otras variantes.
3. `entrenar.py` inicializa MobileNetV2 alfa 0,35 con pesos generales de ImageNet. Congela la base y ajusta una salida nueva durante 40 épocas. No carga el modelo bovino previo ni lee las imágenes de prueba durante el entrenamiento.
4. Desde la raíz del proyecto, ejecutar `.venv/Scripts/python.exe 04_evaluacion/evaluar.py --config experimentos/evaluacion_20261016/config_experimento.json`.

Los scripts de preparación y entrenamiento se detienen si ya existe su salida para conservar la ejecución original. Para repetir el experimento, utilizar una carpeta nueva y conservar las mismas entradas. La configuración generada contiene rutas absolutas del equipo y se regenera en la preparación.

## Interpretación

La prueba contiene imágenes que no participaron en el ajuste de esta versión. La agrupación controla las copias y variantes identificadas; no certifica independencia por animal, sesión o establecimiento. Por tanto, los resultados son una evaluación interna de imágenes y no una validación clínica externa. Las etiquetas son las proporcionadas con las colecciones.

La procedencia ocular se cotejó con los archivos locales de Roboflow y sus declaraciones de dominio público. La descripción inicial del autor identifica Mendeley Lumpy Skin Images Dataset y un conjunto LSD de Kaggle para las clases de piel. La ficha https://data.mendeley.com/datasets/w36hpf86j2/1 declara CC BY 4.0; esa licencia no se extiende a las imágenes de la otra fuente sin identificar. Las muestras de piel no conservan el desglose por fuente. La clase no válida no incorpora metadatos de fuente o licencia.

## Evidencia conservada

- `conteos.json`: entradas, imágenes únicas, exclusiones y distribución por clase.
- `division/manifiesto.json`: imágenes, huellas, grupos y conjuntos.
- `trazabilidad_muestras.json`: correspondencias con las fuentes locales.
- `entrenamiento.json`: configuración y evolución del ajuste.
- `resultados/`: métricas, predicciones individuales y matrices producidas por `evaluar.py`.
- `modelo/`: nueva versión experimental con sus cinco etiquetas.

El informe Word se construye con `docs/crear_informe_evaluado.py`, que toma sus cifras directamente de los resultados y comprueba la extensión de 5.000 a 5.500 palabras.
