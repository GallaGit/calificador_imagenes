# Stack tecnologico

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

## Procesamiento de imagenes

- Pillow: validacion de archivos, lectura de dimensiones y creacion de muestras.
- Matplotlib: figura de exploracion con nueve gatos y nueve perros.
- `ImageDataGenerator` y `flow_from_directory`: lectura progresiva, normalizacion y redimensionado de imagenes.

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
