# Estado actual

## Fases completadas

### Fase 1: comprender y explorar

Completada y verificada con el dataset real.

- Script: `src/explore_dataset.py`.
- Dataset validado: 25.000 imagenes.
- Sin archivos invalidos.
- Commit: `a8642b0 feat: explore dogs and cats dataset`.

### Fase 2: preparar imagenes

Completada y verificada con TensorFlow/Keras.

- Script: `src/prepare_dataset.py`.
- Division 80/20 creada.
- Generadores validados con batches de forma `(32, 224, 224, 3)`.
- Commit principal: `9a8d7c8 feat: prepare image generators`.

### Configuracion del entorno

Completada.

- Python 3.11 instalado en paralelo.
- TensorFlow 2.21.0 instalado en `C:\venvs\calificador_imagenes`.
- Cursor configurado para usar el interprete del proyecto.
- Commits: `25fbe93`, `5011251` y `1721281`.

## Fase siguiente

La Fase 3 consiste en construir y compilar la RNA basada en EfficientNet-B0. Todavia no se ha creado el modelo ni se ha iniciado el entrenamiento.

## Restricciones conocidas

- La RTX 4060 no esta disponible para TensorFlow en Windows nativo.
- El entrenamiento inicial se ejecutara en CPU, posiblemente con pocas epocas para validar el flujo.
- WSL2/CUDA queda como optimizacion de infraestructura, no como requisito para escribir o probar el modelo.

## Comandos utiles

Explorar el dataset:

```powershell
C:\venvs\calificador_imagenes\Scripts\python.exe src\explore_dataset.py
```

Preparar y validar los generadores:

```powershell
C:\venvs\calificador_imagenes\Scripts\python.exe src\prepare_dataset.py
```

Comprobar TensorFlow y GPU:

```powershell
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```
