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

## Decisiones iniciales pendientes

- Confirmar la ruta exacta del dataset descargado.
- Usar `224 x 224` como tamano final para coincidir con EfficientNet-B0.
- Dividir los datos en 80% para entrenamiento y 20% para prueba.
- Confirmar si el entrenamiento se realizara con GPU o CPU.
