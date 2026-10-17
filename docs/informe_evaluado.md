@RESUMEN
El proyecto desarrolla una aplicación web para clasificar fotografías de piel y ojos bovinos en cinco categorías: piel sana, dermatosis nodular contagiosa, imagen no válida, queratoconjuntivitis infecciosa bovina y ojo sano. Su propósito es apoyar la observación del ganado mediante una interfaz que permite cargar una fotografía, consultar su clasificación y recibir una orientación. El trabajo se organizó con CRISP-DM y se implementó con Python, TensorFlow, Streamlit y Pillow. Para obtener resultados propios, se revisaron las cinco colecciones de imágenes, se trataron las repeticiones y se reservaron 30 fotografías por clase antes del entrenamiento. Además, se excluyeron las imágenes oculares identificadas como procedentes del conjunto de cabras. Se entrenó una nueva versión con transferencia de aprendizaje desde ImageNet y se ejecutó el programa de evaluación del proyecto. En las 150 imágenes de prueba se obtuvo una exactitud de {{EXACTITUD}}, una precisión macro de {{PRECISION}} y una sensibilidad macro de {{SENSIBILIDAD}}. Estos resultados describen una evaluación interna de clasificación y no una validación clínica. El siguiente paso es contrastar el sistema con fotografías de nuevos animales y condiciones de campo, con etiquetas revisadas por profesionales veterinarios. La aplicación integra el análisis visual, los controles de calidad y una presentación comprensible de los resultados.
@PALABRAS inteligencia artificial, bovinos, clasificación de imágenes, transferencia de aprendizaje, aplicación web
@CUERPO
El proyecto aplica la clasificación automática de imágenes a la observación de piel y ojos bovinos. Reúne un modelo de aprendizaje supervisado y una aplicación web en español, de modo que el usuario pueda consultar una fotografía sin ejecutar directamente las herramientas de entrenamiento.

El informe presenta los objetivos, la preparación de datos, el funcionamiento y los resultados del experimento. El producto es un prototipo funcional con una evaluación interna reproducible. Las conclusiones se apoyan en sus predicciones y distinguen el rendimiento de clasificación del diagnóstico veterinario.

# Planteamiento del problema
La observación del ganado puede apoyarse en fotografías, pero la utilidad de una imagen depende de su calidad y de su contexto. La iluminación, el encuadre, la distancia y el movimiento del animal modifican el aspecto de una región corporal. Una fotografía borrosa o demasiado oscura puede dificultar tanto la revisión humana como el procesamiento automático. Por ello, el sistema necesita controlar la entrada y ofrecer una forma sencilla de repetir la captura.

El problema abordado consiste en integrar un clasificador de imágenes en una herramienta que presente respuestas comprensibles. La pregunta de desarrollo es cómo analizar fotografías de piel y ojos bovinos, relacionarlas con cinco categorías y orientar al usuario cuando la imagen o la clasificación no resulten suficientes. La aplicación se centra en la semejanza visual con las clases aprendidas y en la presentación de esa información.

El proyecto tiene valor académico porque conecta los contenidos de Inteligencia Artificial con decisiones concretas de software. Incluye preparación de datos, aprendizaje supervisado, separación de conjuntos, medición del rendimiento y diseño de una interfaz. La experiencia también permite reconocer que una salida numérica necesita una interpretación: una categoría puede representar una alerta visual sin identificar por sí sola la causa de un problema.

# Objetivos
## Objetivo general
Desarrollar y evaluar una aplicación web que clasifique fotografías de piel y ojos bovinos mediante aprendizaje supervisado, presente el resultado de manera comprensible y apoye la observación del ganado dentro del alcance definido para el proyecto.

## Objetivos específicos
Organizar las imágenes de las cinco clases y revisar su cantidad, sus repeticiones y su procedencia. Preparar una separación reproducible que reserve 30 imágenes por clase antes del entrenamiento. Integrar el modelo con una interfaz que permita cargar fotografías y consultar información relacionada con su funcionamiento.

Medir la exactitud, la precisión, la sensibilidad y la matriz de confusión del modelo entrenado para el experimento. Incorporar controles de archivo, calidad y decisión que regulen la presentación de resultados. Finalmente, interpretar los hallazgos y establecer una proyección de validación con nuevas imágenes y condiciones de uso.

# Fundamentos del proyecto
## Clasificación de imágenes y aprendizaje supervisado
En el aprendizaje supervisado, cada imagen se relaciona con una etiqueta conocida. El entrenamiento ajusta los parámetros del clasificador para que reconozca características asociadas a esas etiquetas. En una tarea multiclase, el modelo debe elegir entre varias categorías. Aquí se utilizan cinco: dos relacionadas con piel, dos con ojos y una categoría de rechazo para imágenes ajenas al objetivo.

Una red neuronal no consulta una ficha veterinaria al recibir una fotografía. Procesa valores de píxeles y produce puntuaciones a partir de patrones aprendidos. Estos patrones pueden incluir formas, texturas y colores, pero también elementos del fondo o del encuadre. La preparación del conjunto de datos influye, por tanto, en lo que aprende el modelo y en las situaciones en que sus resultados pueden ser útiles.

La clase de imagen no válida se incorporó para representar ejemplos que no corresponden a las regiones de interés. Esto ayuda a evitar que toda fotografía termine forzosamente presentada como una condición bovina. Sin embargo, esa clase solo representa la diversidad incluida en sus ejemplos; su presencia no garantiza el reconocimiento de cualquier objeto o situación desconocida.

## Transferencia de aprendizaje
La transferencia de aprendizaje aprovecha una red entrenada previamente para extraer características visuales. En lugar de ajustar todos sus parámetros desde valores aleatorios, se utiliza esa base y se entrena una salida adaptada al nuevo problema. TensorFlow describe este procedimiento como una forma de reutilizar representaciones aprendidas y construir clasificadores con conjuntos de datos más pequeños (TensorFlow, s. f.).

Para el experimento se utilizó MobileNetV2, una arquitectura que combina bloques residuales invertidos y convoluciones eficientes (Sandler et al., 2018). Se eligió la variante con multiplicador de anchura de 0,35 y pesos iniciales de ImageNet. La base se mantuvo congelada y se ajustó una nueva capa de clasificación de cinco salidas. Esta elección permitió ejecutar el entrenamiento en el equipo disponible sin convertir la capacidad de cómputo en el centro del proyecto.

La versión inicial de la aplicación se había construido alrededor de una exportación de Teachable Machine. La evaluación de este informe corresponde a una nueva versión experimental entrenada después de separar la prueba. No se continuó el entrenamiento desde los pesos bovinos anteriores, porque esas imágenes ya habían participado en su aprendizaje. Esta distinción permite vincular los resultados con un procedimiento concreto.

## Contexto de las clases
La dermatosis nodular contagiosa es una enfermedad viral del ganado cuyo cuadro puede incluir lesiones nodulares en la piel. La Organización Mundial de Sanidad Animal describe su relevancia sanitaria y las medidas de prevención y control a escala animal y poblacional (World Organisation for Animal Health, 2026). En el proyecto, su nombre identifica una clase visual de referencia, sin que una fotografía sustituya los procedimientos necesarios para confirmar la enfermedad.

La queratoconjuntivitis infecciosa bovina puede presentar signos como lagrimeo, dolor ocular y alteraciones corneales. Estos cambios motivan la consulta de la región ocular, aunque otros problemas pueden compartir una apariencia semejante (Angelos, 2024). Por eso, la aplicación utiliza el nombre de la enfermedad como categoría del modelo y acompaña el resultado con una orientación de revisión veterinaria.

Las clases piel sana y ojo sano describen el aspecto representado en sus imágenes de referencia. Una predicción de estas categorías se refiere únicamente a la fotografía analizada. El sistema no evalúa todo el animal ni considera temperatura, comportamiento, historia clínica o pruebas complementarias. Esta definición mantiene la interpretación dentro de la información que realmente recibe el modelo.

# Metodología
El desarrollo se organizó con CRISP-DM, que relaciona la comprensión del problema, la comprensión de los datos, su preparación, el modelado, la evaluación y la puesta en uso. Estas fases permiten revisar decisiones anteriores a medida que aparecen nuevos hallazgos (IBM, s. f.). En el proyecto, la revisión de las imágenes oculares produjo un ajuste del conjunto experimental antes del entrenamiento. La Tabla 1 resume la aplicación de esta metodología.

@TABLA fases

La evaluación se realizó con el mismo componente de clasificación utilizado por la aplicación, para reducir diferencias entre la medición y el funcionamiento del programa.

Se conservaron los archivos originales y se trabajó con copias. Cada imagen seleccionada quedó asociada con su clase, su conjunto y una huella digital. Este registro permite comprobar la división utilizada, detectar modificaciones accidentales y facilitar la repetición del procedimiento.

# Datos y preparación
## Inventario y procedencia
Las cinco colecciones entregadas contienen {{TOTAL_ENTRADAS}} imágenes. La revisión comprobó que los archivos pueden abrirse y que las muestras tienen dimensiones de 224 por 224 píxeles. El recuento actual de piel sana es 782. Las 853 imágenes mencionadas anteriormente procedían de la descripción inicial del proyecto. Para esta versión se utiliza el contenido efectivamente analizado, con una diferencia de 71 imágenes respecto de aquel dato. Así, el informe mantiene una cifra coherente con el archivo de muestras actual.

La Tabla 2 recoge la procedencia que puede sostenerse con los archivos disponibles. En las clases de ojos se encontraron coincidencias con dos exportaciones de Roboflow, llamadas sapipise y kambingpise. Sus archivos descriptivos identifican al responsable del conjunto y declaran dominio público. La correspondencia se comprobó comparando las imágenes, además de revisar sus nombres. Las muestras asociadas al conjunto de cabras se excluyeron del experimento bovino.

@TABLA procedencia

Para las clases de piel, la descripción inicial identifica Lumpy Skin Images Dataset de Mendeley y otro conjunto de LSD de Kaggle. La ficha de Mendeley declara 324 imágenes de piel afectada, 700 de piel normal y licencia CC BY 4.0 (Kumar & Shastri, 2022). Esas cantidades describen la fuente pública, no la colección combinada del proyecto. Los archivos actuales no conservan el desglose por fuente ni el enlace específico de Kaggle. En imágenes no válidas tampoco consta una fuente original. La tabla distingue los datos comprobados de la información no documentada, sin extender una licencia a toda la colección.

El inventario inicial tampoco equivale al número de fotografías independientes. Algunas entradas repiten exactamente los mismos píxeles y otras representan modificaciones de una fotografía. Además, las colecciones oculares incluyen aumentos como giros y cambios de iluminación. Contar esas variantes como observaciones completamente distintas puede exagerar la diversidad del conjunto si se distribuyen sin control entre entrenamiento y prueba.

## Revisión de duplicados y selección
La preparación combinó dos comprobaciones. La primera identificó imágenes idénticas mediante una huella de sus píxeles. La segunda buscó semejanzas visuales con una huella perceptual de 64 bits y agrupó las coincidencias próximas. Para las imágenes relacionadas con Roboflow también se utilizaron las familias de nombres que conectan las versiones aumentadas de una misma fotografía. Así se evitó repartir variantes identificadas entre ambos conjuntos.

El umbral de similitud perceptual se fijó en una distancia máxima de cuatro bits. Esta regla sirve para detectar posibles copias cercanas, pero no identifica por sí sola a un animal. Por ese motivo, el procedimiento se presenta como agrupación de imágenes y variantes conocidas. Cuando un grupo reunía etiquetas diferentes, se excluyó del experimento para no resolver arbitrariamente cuál era la categoría correcta.

La comparación con los originales oculares incluyó un control adicional de diferencia media entre píxeles después de ajustar el tamaño. Se aceptaron correspondencias con una diferencia media no superior a ocho unidades en la escala de 0 a 255. Las fotografías identificadas como procedentes de kambingpise se retiraron del conjunto bovino. Esta revisión conserva el objetivo del proyecto y evita mezclar especies bajo nombres clínicos específicos de bovinos.

La Tabla 3 distingue entradas recibidas, imágenes únicas, entrenamiento y prueba. Las diferencias entre columnas se explican por las repeticiones, las exclusiones y las variantes de los grupos reservados. Los archivos entregados se conservaron; la depuración se aplicó a las copias preparadas para el experimento.

@TABLA distribucion

## Separación previa al entrenamiento
Se reservaron 30 imágenes por clase, para un total de 150 fotografías de prueba. La selección utilizó una semilla fija de 42 y se realizó antes de entrenar. Cada imagen de prueba procede de un grupo distinto dentro de su clase. Las demás variantes identificadas de esos grupos quedaron fuera del entrenamiento, mientras que los grupos restantes aportaron los ejemplos utilizados para ajustar la nueva salida del modelo.

Esta separación responde al problema de evaluar con las mismas imágenes usadas para aprender. Un resultado sobre imágenes de entrenamiento mide cuánto se ajusta el clasificador a esos ejemplos. La prueba reservada permite observar cómo responde a imágenes que no participaron en el ajuste de esta versión. Aun así, la evaluación sigue siendo interna, porque las imágenes proceden de las colecciones disponibles y no de una campaña nueva de captura.

No se utilizaron las imágenes de prueba para elegir la cantidad de épocas, cambiar los umbrales o seleccionar entre varios modelos. El experimento siguió una configuración fijada de antemano y se midió al finalizar. La independencia por animal, establecimiento y sesión de captura pertenece al diseño de una validación de campo, donde esas variables podrán registrarse desde la obtención de cada fotografía.

# Entrenamiento e integración
## Preparación de la entrada
Antes de clasificar, la imagen se convierte a RGB y se ajusta a 224 por 224 píxeles mediante un recorte central. Los valores se expresan como números de punto flotante y se normalizan al intervalo de menos uno a uno. La implementación inicial tomó como referencia el ejemplo oficial de Google Creative Lab (s. f.) para imágenes exportadas desde Teachable Machine. En el nuevo entrenamiento se reutilizó exactamente la función de preparación de la aplicación.

Pillow se utiliza para abrir los archivos, corregir su orientación, convertir sus canales y preparar la imagen (Pillow contributors, s. f.). Mantener estas operaciones en una función compartida evita que el entrenamiento reciba una transformación y la interfaz aplique otra. La coherencia del procesamiento es necesaria para interpretar una evaluación como evidencia del comportamiento del clasificador integrado.

## Configuración del experimento
La base MobileNetV2 se cargó con pesos de ImageNet y permaneció congelada. Sobre sus características se entrenó una capa densa de cinco salidas con activación softmax. Se incorporó una tasa de abandono de 0,20 y regularización L2 de 0,0001. El ajuste se realizó durante 40 épocas, con lotes de 32, el optimizador Adam y una tasa de aprendizaje de 0,001. Estos valores describen el experimento realizado; no representan una búsqueda del mejor conjunto posible de parámetros.

Para compensar las diferencias de cantidad entre clases se asignaron pesos inversamente proporcionales a su frecuencia en entrenamiento. Así, una clase con menos ejemplos tuvo una contribución mayor por imagen a la función de pérdida. Esta medida equilibra el ajuste, aunque no crea diversidad adicional ni sustituye la incorporación de nuevos ejemplos. La Tabla 4 reúne los parámetros principales del entrenamiento y de la aplicación.

@TABLA parametros

El modelo resultante se exportó en un formato compatible con el clasificador del proyecto, conservando el orden de las cinco etiquetas. La versión experimental se almacenó junto con su configuración y sus resultados, separada de la exportación anterior de Teachable Machine. De esta forma, los números del informe se refieren a un modelo identificable y no se atribuyen automáticamente a versiones distintas.

# Funcionamiento de la aplicación web
La interfaz ofrece dos áreas principales. La primera permite cargar una imagen o realizar una captura puntual con la cámara del navegador y consultar el resultado. La segunda presenta información relacionada con el modelo y sus clases. Streamlit permite construir estos controles y mostrar imágenes y resultados desde una aplicación escrita en Python (Streamlit, s. f.). El usuario interactúa desde el navegador mientras el servidor realiza la inferencia.

La Figura 1 resume el recorrido de una fotografía. El sistema comprueba el archivo, prepara la imagen, revisa su calidad y obtiene las puntuaciones del modelo. Después aplica las reglas de decisión y presenta la categoría correspondiente. Si la entrada no reúne las condiciones definidas, solicita otra imagen. La evaluación utiliza el mismo clasificador, aunque calcula las métricas sobre todas las imágenes de prueba para no esconder los casos difíciles.

@FIGURA flujo

Una categoría de enfermedad motiva una orientación de revisión profesional; una categoría saludable describe el aspecto fotografiado. La clase no válida solicita una imagen pertinente. Cuando las puntuaciones no cumplen las reglas de aceptación, el sistema se abstiene de ofrecer un resultado firme.

La regla de aceptación considera una puntuación mínima de 0,70 y una diferencia mínima de 0,15 entre las dos categorías principales. Estas cantidades son parámetros de funcionamiento. No son medidas del rendimiento obtenido en la prueba ni porcentajes clínicos de enfermedad. La interfaz también permite consultar las puntuaciones para entender qué categorías compiten en una imagen concreta.

Los controles de calidad revisan dimensiones, brillo y textura. Se solicita una región visible, buena iluminación y una fotografía por zona de interés. Una imagen aceptada por estos controles todavía puede clasificarse incorrectamente, por lo que su efecto se analiza de forma separada.

La aplicación web puede alojarse en un servidor que disponga del modelo y sus dependencias. El componente adicional de cámara continua utiliza procesamiento local y reglas temporales de alertas; su prueba en campo se incluye en la proyección. La captura puntual del navegador constituye el flujo web descrito en este informe.

# Evaluación y resultados
## Procedimiento y métricas
La evaluación se ejecutó con evaluar.py después del entrenamiento. El programa verificó los archivos de la división, cargó el modelo experimental y calculó una predicción por imagen. Las métricas se obtuvieron con scikit-learn, que permite calcular precisión, sensibilidad, F1 y soportes por clase (scikit-learn developers, s. f.). Se conservaron tanto los resultados agregados como las predicciones individuales para revisar los errores.

La exactitud expresa la proporción total de imágenes clasificadas correctamente. La precisión de una clase indica qué proporción de las imágenes asignadas a esa categoría tenía realmente esa etiqueta. La sensibilidad, también llamada exhaustividad, indica qué proporción de los ejemplos reales de una clase fue reconocida. F1 resume el equilibrio entre precisión y sensibilidad. En todos los casos, la comparación se realiza frente a las etiquetas disponibles en el conjunto experimental.

El promedio macro asigna la misma importancia a cada clase. Esto permite observar el comportamiento conjunto sin dar más peso a la categoría con mayor cantidad de imágenes. Como la prueba contiene 30 ejemplos de cada clase, la sensibilidad macro coincide con la exactitud en este experimento. La matriz de confusión complementa esos valores mostrando hacia qué categoría se dirige cada error.

## Resultados del modelo
{{RESULTADOS_TEXTO}}

@TABLA metricas

La Figura 2 presenta la matriz de confusión de las 150 imágenes reservadas. Cada fila corresponde a una etiqueta real y cada columna a una predicción. Los valores de la diagonal son aciertos; los demás representan confusiones. La suma de cada fila es 30, lo que permite comparar directamente el número de errores entre clases sin ajustar por diferencias de tamaño.

{{ERRORES_TEXTO}}

@FIGURA matriz

## Resultados con las reglas de la aplicación
{{COBERTURA_TEXTO}}

La distinción entre clasificación y aceptación es relevante para el uso. Una abstención puede evitar presentar una respuesta débil, pero obliga a repetir la fotografía o buscar otra forma de valoración. Por ello, una evaluación de la interfaz debe considerar simultáneamente la calidad de las respuestas aceptadas y la frecuencia con que se obtiene una respuesta. Mostrar solo la exactitud de los casos aceptados daría una visión incompleta.

## Comprobaciones de funcionamiento
El proyecto dispone de pruebas automatizadas para la carga del modelo, la consistencia de etiquetas, las reglas de decisión, el tratamiento de imágenes y los controles de seguridad. En la revisión funcional registrada se completaron 28 pruebas satisfactorias. Estas comprobaciones respaldan el comportamiento del software ante situaciones definidas. Las métricas de clasificación se obtuvieron por separado, mediante el experimento con fotografías descrito anteriormente.

# Discusión
{{DISCUSION_BRECHA}}

La depuración mostró que la cantidad de archivos no representa necesariamente la cantidad de observaciones distintas. En piel con dermatosis se identificaron numerosas repeticiones exactas. En los ojos aparecieron variantes de fotografías y material de dos especies. Estos hallazgos justifican tratar la comprensión de los datos como parte del desarrollo del modelo, porque una división realizada sin esa revisión podría producir una estimación demasiado favorable.

Los errores también deben interpretarse según el propósito de cada clase. Confundir una imagen ocular alterada con ojo sano puede reducir la atención que recibe esa fotografía. Confundir una imagen saludable con una clase de enfermedad puede generar una consulta innecesaria. El prototipo no asigna un costo clínico numérico a estas situaciones, pero la matriz permite reconocerlas y orientar la selección de ejemplos para una siguiente evaluación.

La separación de prueba reduce la reutilización directa de ejemplos en el ajuste, aunque no demuestra independencia completa entre animales. Las colecciones no proporcionan una identificación individual suficiente para establecerla en todos los casos. Por eso se informa una evaluación interna de imágenes. El paso hacia una afirmación de generalización requiere nuevas capturas cuyo origen, animal y sesión puedan distinguirse desde el inicio.

El tamaño de prueba facilita una primera comparación equilibrada: cada clase tiene la misma cantidad de ejemplos. Sin embargo, con 30 fotografías por clase, un único acierto o error modifica su sensibilidad en aproximadamente 3,33 puntos porcentuales. Diferencias pequeñas entre categorías deben interpretarse con esa resolución. Los resultados no autorizan a afirmar que el mismo porcentaje se repetirá en cualquier establecimiento o dispositivo.

La clasificación puede verse favorecida por características ajenas a una lesión, como fondos, encuadres o estilos de fotografía propios de una fuente. La revisión visual mostró que la clase de piel sana contiene tanto vistas amplias del animal como imágenes más cercanas. En futuras capturas conviene controlar esa diversidad y representar condiciones comparables en las clases de interés, de manera que la tarea dependa más de la región observada.

Las reglas de aceptación aportan una capa adicional de interpretación. Con las puntuaciones normalizadas de cinco clases, exigir 0,70 para la primera categoría ya implica una separación amplia respecto de la segunda. En esta configuración, el margen de 0,15 no añade una restricción práctica cuando se cumple ese umbral. Su utilidad puede revisarse si cambia el umbral, manteniendo cualquier ajuste separado de una futura prueba final.

El experimento establece una referencia reproducible para comparar posteriormente otras configuraciones con la misma disciplina de separación y evaluación. La elección de MobileNetV2 permite documentar una solución concreta, sin afirmar que sea la mejor arquitectura para esta tarea.

# Seguridad y privacidad
El análisis de imágenes se ejecuta con el modelo cargado en el servidor y no necesita una clave de una API externa de inteligencia artificial. Cuando la aplicación se publica, la fotografía viaja desde el navegador hasta ese servidor para ser procesada. Esta descripción aclara dónde ocurre el análisis y permite informar al usuario sobre el tratamiento de sus imágenes.

La carga está limitada por formato, tamaño y cantidad de píxeles. También se controla la apertura de archivos y se eliminan los metadatos de la representación utilizada para mostrar y analizar la imagen. Estas decisiones reducen exposiciones innecesarias, como conservar información de ubicación incluida en una fotografía. El flujo ordinario no guarda deliberadamente las imágenes cargadas como un archivo permanente de consultas.

La revisión técnica registrada no encontró credenciales expuestas dentro del alcance examinado. El servidor incorpora controles frente a solicitudes cruzadas y oculta detalles internos de errores. La seguridad de una publicación también depende del alojamiento, la administración del acceso y la actualización de dependencias.

# Alcance
El sistema clasifica una fotografía entre cinco categorías, muestra sus puntuaciones y aplica controles para decidir si presenta un resultado o solicita otra captura. Su ámbito es el apoyo visual académico para piel y ojos bovinos. La unidad analizada es una imagen de una región visible, con las condiciones de entrada descritas en el informe.

La aplicación no localiza automáticamente cada lesión, no identifica todas las enfermedades posibles y no integra antecedentes clínicos del animal. Sus etiquetas representan las categorías aprendidas. La evaluación realizada mide la clasificación interna frente a esas etiquetas y no sustituye una validación diagnóstica. Esta delimitación permite presentar el producto con un propósito concreto y mantener una interpretación coherente de sus resultados.

# Proyección
La siguiente etapa consiste en reunir fotografías de nuevos animales, establecimientos y dispositivos, con revisión veterinaria de las etiquetas. El registro de animal y sesión permitirá separar las observaciones desde la captura y analizar la respuesta bajo condiciones de iluminación, distancia y movimiento propias del campo. La evaluación mantendrá una prueba final que no participe en el ajuste de parámetros.

También se propone ampliar progresivamente las enfermedades representadas, fortalecer la variedad de imágenes no válidas y completar la documentación de procedencia de las colecciones. La cámara continua podrá probarse con usuarios en condiciones reales, evaluando tanto la clasificación como la estabilidad de las alertas y la facilidad para obtener una fotografía útil. Estas líneas extienden el prototipo a partir de la referencia experimental presentada.

# Conclusiones
Se desarrolló una aplicación web que integra carga de fotografías, clasificación de cinco categorías, controles de calidad y presentación de resultados en español. La organización del proyecto permitió conectar la interfaz con un componente de inferencia compartido y ejecutar una evaluación sobre el modelo preparado para el experimento.

La revisión de datos corrigió el recuento de piel sana a 782 imágenes, identificó repeticiones y separó del experimento las fotografías oculares asociadas al conjunto de cabras. Tras reservar 30 imágenes por clase, se entrenó una nueva versión desde una base general y se evaluaron 150 imágenes que no participaron en su ajuste. La exactitud fue {{EXACTITUD}}, la precisión macro {{PRECISION}} y la sensibilidad macro {{SENSIBILIDAD}}. Estas cifras corresponden a esa versión experimental y a la división descrita.

El trabajo demuestra la integración técnica y aporta una referencia cuantitativa interna para continuar el desarrollo. El siguiente paso de validación es medir el comportamiento con imágenes de nuevos animales y condiciones de campo, revisadas por profesionales veterinarios. Hasta entonces, el alcance de los resultados es la clasificación de las imágenes evaluadas y su uso como apoyo visual dentro del prototipo.

@REFERENCIAS
Angelos, J. A. (2024, abril). |Infectious keratoconjunctivitis in cattle and small ruminants|. MSD Veterinary Manual. https://www.msdvetmanual.com/eye-diseases-and-disorders/infectious-keratoconjunctivitis/infectious-keratoconjunctivitis-in-cattle-and-small-ruminants
Google Creative Lab. (s. f.). |Teachable Machine image model TensorFlow and Keras example| [Código de ejemplo]. https://github.com/googlecreativelab/teachablemachine-community/blob/master/snippets/markdown/image/tensorflow/keras.md
IBM. (s. f.). |CRISP-DM help overview|. IBM SPSS Modeler. https://www.ibm.com/docs/en/spss-modeler/18.6.0?topic=dm-crisp-help-overview
Irawan, M. A. A. (2023a). |Kambingpise| (Versión 1) [Conjunto de datos]. Roboflow Universe. https://universe.roboflow.com/muhammad-athar-althariq-irawan-m346dkx3922-wmdko/kambingpise
Irawan, M. A. A. (2023b). |Sapipise| (Versión 1) [Conjunto de datos]. Roboflow Universe. https://universe.roboflow.com/muhammad-athar-althariq-irawan-m346dkx3922-wmdko/sapipise
Kumar, S., & Shastri, S. (2022). |Lumpy Skin Images Dataset| (Versión 1) [Conjunto de datos]. Mendeley Data. https://doi.org/10.17632/w36hpf86j2.1
Pillow contributors. (s. f.). |Image module|. Pillow documentation. https://pillow.readthedocs.io/en/stable/reference/Image.html
Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). |MobileNetV2: Inverted residuals and linear bottlenecks|. arXiv. https://doi.org/10.48550/arXiv.1801.04381
scikit-learn developers. (s. f.). |Classification report|. scikit-learn 1.7.2 documentation. https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.classification_report.html
Streamlit. (s. f.). |st.image|. Streamlit documentation. https://docs.streamlit.io/develop/api-reference/media/st.image
TensorFlow. (s. f.). |Transfer learning and fine-tuning|. https://www.tensorflow.org/tutorials/images/transfer_learning
World Organisation for Animal Health. (2026, 9 de julio). |Lumpy skin disease|. https://www.woah.org/en/disease/lumpy-skin-disease/
