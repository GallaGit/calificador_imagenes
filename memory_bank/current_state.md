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

### Fase 3, paso 1: crear el modelo

Completada y verificada.

- Script: `src/build_model.py`.
- Arquitectura: EfficientNet-B0, GlobalAveragePooling2D y dos capas Dense.
- Entrada: `(224, 224, 3)`.
- Salida: `(2,)`, con probabilidades para `cat` y `dog`.
- Este paso no entrena el modelo.

## Fases 3 a 6: implementadas

- `src/train_model.py` ejecuta el entrenamiento inicial, validacion, checkpoint,
  early stopping, evaluacion, matriz de confusion y exportacion.
- `src/predict.py` y `src/app.py` permiten reutilizar el modelo con una imagen.
- Los resultados se guardan en `models/` y `artifacts/`.
- El entrenamiento real depende de ejecutar el script con TensorFlow y el dataset
  preparado; no se ejecuta automaticamente porque puede tardar varias horas en CPU.

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
