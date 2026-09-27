# Stack tecnologico

Este documento describe las tecnologias que participan en el proyecto y la
responsabilidad de cada una dentro del flujo de trabajo.

## Python

- Python global del equipo: `3.14.6`.
- Python usado por el proyecto: `3.11.9`.
- Motivo: TensorFlow no dispone de una instalacion compatible con el Python global 3.14 en este entorno.
- Entorno aislado: `C:\venvs\calificador_imagenes`.
- El entorno aislado no se versiona en Git.

Comando recomendado para ejecutar scripts:

```powershell
C:\venvs\calificador_imagenes\Scripts\python.exe src\prepare_dataset.py
```

En Cursor, el workspace esta configurado para priorizar ese interprete en las terminales nuevas.

## TensorFlow y Keras

- TensorFlow: `2.21.0`.
- Keras se utiliza a traves de `tensorflow.keras`.
- Motivo: TensorFlow ejecuta los calculos de la red neuronal y Keras ofrece una API sencilla para crear modelos, generadores, callbacks y evaluaciones.

### Componentes de TensorFlow/Keras utilizados

- `EfficientNetB0`: red convolucional usada como extractor de caracteristicas
  visuales. Se utiliza sin la cabeza original de ImageNet (`include_top=False`)
  para adaptarla a las clases `cat` y `dog`.
- `Sequential`: organiza las capas del clasificador en orden.
- `GlobalAveragePooling2D`: convierte los mapas de caracteristicas de la red
  convolucional en un vector compacto.
- `Dense`: capas totalmente conectadas que aprenden la decision final. El
  proyecto usa `Dense(128, relu)` y `Dense(2, softmax)`.
- `Adam`: optimizador que ajusta los pesos de la red durante el entrenamiento.
- `categorical_crossentropy`: funcion de perdida compatible con las etiquetas
  one-hot generadas para las dos clases.
- `ImageDataGenerator`: normaliza los pixeles y reserva el 10% del entrenamiento
  para validacion.
- `flow_from_directory`: carga las imagenes por batches desde las carpetas sin
  cargar todo el dataset en memoria.
- `ModelCheckpoint`: conserva el modelo con mejor `val_accuracy`.
- `EarlyStopping`: detiene el entrenamiento cuando `val_loss` deja de mejorar y
  restaura los mejores pesos.
- `tf.keras.models.load_model`: carga el checkpoint para la evaluacion y la
  prediccion posterior.

## Procesamiento de imagenes

- Pillow (`PIL`): valida que los archivos sean imagenes, lee dimensiones,
  convierte imagenes a RGB y redimensiona la imagen recibida para prediccion.
- Matplotlib: genera muestras del dataset, graficas de perdida/accuracy y la
  matriz de confusion.
- NumPy: transforma batches y probabilidades en arrays, y obtiene la clase con
  mayor probabilidad.

## Evaluacion y resultados

- scikit-learn: calcula el reporte de clasificacion y la matriz de confusion a
  partir de las etiquetas reales y las predicciones.
- JSON de la biblioteca estandar: guarda el historial de entrenamiento y los
  metadatos del modelo, incluyendo tamano de entrada, clases y metricas.

## Herramientas de Python y sistema

- Python `3.11.9`: lenguaje principal del proyecto y version usada para mantener
  compatibilidad con TensorFlow en este entorno.
- `venv` ubicado en `C:\venvs\calificador_imagenes`: entorno aislado para que
  las dependencias del proyecto no modifiquen Python global.
- `argparse`: permite configurar rutas, epochs, batch size, paciencia y semilla
  desde la terminal.
- `pathlib`: gestiona rutas de archivos de forma portable.
- Git: controla versiones del codigo y excluye datasets, modelos y artefactos
  pesados mediante `.gitignore`.
- VS Code/Cursor y `.vscode/settings.json`: seleccionan el interprete Python del
  proyecto y facilitan la ejecucion de los scripts.

## Archivos principales y tecnologias relacionadas

| Archivo | Tecnologias | Responsabilidad |
| --- | --- | --- |
| `src/explore_dataset.py` | Pillow, Matplotlib, `pathlib` | Valida nombres, formatos, dimensiones y muestras del dataset. |
| `src/prepare_dataset.py` | Python estandar, Keras | Divide las imagenes y crea los generadores. |
| `src/build_model.py` | TensorFlow/Keras, EfficientNet-B0 | Construye y verifica la arquitectura. |
| `src/train_model.py` | TensorFlow/Keras, Matplotlib, NumPy, scikit-learn | Entrena, optimiza, evalua y exporta el modelo. |
| `src/predict.py` | TensorFlow/Keras, Pillow, NumPy | Carga el modelo y clasifica una imagen. |
| `src/app.py` | Python | Punto de entrada de la prediccion. |

## Dependencias de plantilla no usadas en el clasificador

`requirements.txt` conserva algunas dependencias heredadas de la plantilla de
ciencia de datos, como SQLAlchemy, pandas, psycopg2, pymysql, requests,
seaborn, xgboost, sympy, ipyleaflet e ipywidgets. No forman parte del pipeline
actual de entrenamiento ni son necesarias para clasificar imagenes. La funcion
`db_connect` de `src/utils.py` tambien pertenece a esa plantilla y no es usada
por la aplicacion.

## Modelo previsto

- EfficientNet-B0 sin la cabeza superior.
- Entrada: `224 x 224 x 3`.
- `GlobalAveragePooling2D`.
- `Dense(128, activation="relu")`.
- `Dense(2, activation="softmax")`.

## Hardware

- CPU: AMD Ryzen 7 5700X, 8 nucleos y 16 hilos.
- GPU disponible: NVIDIA GeForce RTX 4060.
- Estado actual: TensorFlow en Windows nativo no detecta la GPU y ejecuta en CPU.
- Para usar la RTX 4060 sera necesario configurar WSL2 o Docker con CUDA.

## Dependencias

Las dependencias reproducibles estan declaradas en `requirements.txt`. No se guardan paquetes instalados dentro del repositorio.
