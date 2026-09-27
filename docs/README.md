# Documentacion del proyecto

Este proyecto construye un clasificador de imagenes capaz de distinguir entre perros y gatos usando una red neuronal convolucional basada en EfficientNet-B0.

El trabajo se realizara por fases. No se avanzara a la siguiente fase hasta comprobar la anterior y crear su commit correspondiente.

## Fases

1. [Comprender y explorar el dataset](01-comprender-dataset.md)
2. [Preparar las imagenes](02-preparar-imagenes.md)
3. [Construir la red neuronal](03-construir-rna.md)
4. [Entrenar y evaluar el modelo](04-entrenar-evaluar.md)
5. [Optimizar el modelo](05-optimizar-modelo.md)
6. [Guardar y documentar el resultado](06-guardar-modelo.md)

Las fases 1 a 3 ya fueron implementadas y verificadas antes del entrenamiento:

- Fase 1 valida el dataset real y genera muestras visuales.
- Fase 2 crea la division 80/20 y comprueba batches de Keras.
- Fase 3 construye y compila EfficientNet-B0 y valida la forma de salida.
- Fases 4 a 6 entrenan, optimizan, evalúan y exportan el modelo.

## Stack tecnologico

La referencia completa del stack, las versiones del entorno y la funcion de
cada tecnologia se encuentra en
[`memory_bank/technology_stack.md`](../memory_bank/technology_stack.md).

Resumen del pipeline:

- Python ejecuta los scripts y automatiza el flujo reproducible.
- TensorFlow/Keras construye y entrena la red EfficientNet-B0.
- Pillow y `ImageDataGenerator` preparan las imagenes.
- Matplotlib genera las graficas y la matriz de confusion.
- NumPy y scikit-learn procesan las predicciones y metricas.
- JSON registra metadatos, historial y resultados.
- Git mantiene el codigo versionado y evita subir artefactos pesados.

## Flujo de trabajo

En cada fase seguiremos este orden:

1. Explicar el objetivo con palabras sencillas.
2. Confirmar las decisiones necesarias.
3. Implementar solo los cambios de esa fase.
4. Verificar el resultado.
5. Crear un commit.
6. Pasar a la siguiente fase.

## Estructura de datos prevista

El dataset original se colocara en `data/raw/`. Las imagenes organizadas para Keras se generaran en `data/processed/` y no se subiran al repositorio.

```text
data/
├── raw/
│   └── train/
├── interim/
└── processed/
    ├── train/
    │   ├── cat/
    │   └── dog/
    └── test/
        ├── cat/
        └── dog/
```

## Decisiones confirmadas

- Dataset: `data/raw/train/`.
- Tamano de entrada: `224 x 224 x 3`.
- Division: 80% entrenamiento y 20% prueba.
- Validacion: 10% del conjunto de entrenamiento durante `train_model.py`.
- Hardware: CPU en Windows nativo; la RTX 4060 requiere WSL2 o CUDA configurado.
