# Contexto del proyecto

## Objetivo

Construir un clasificador binario de imagenes que distinga entre gatos y perros usando una red neuronal convolucional basada en EfficientNet-B0.

El dataset procede del conjunto Dogs vs Cats de Kaggle/Asirra. Sus archivos tienen etiquetas en el nombre:

```text
cat.numero.jpg
dog.numero.jpg
```

## Alcance

El proyecto cubre:

1. Exploracion visual y validacion del dataset.
2. Preparacion reproducible de los conjuntos de entrenamiento y prueba.
3. Construccion de una RNA con EfficientNet-B0.
4. Entrenamiento y evaluacion.
5. Optimizacion mediante callbacks.
6. Guardado y documentacion del modelo final.

## Forma de trabajo

El desarrollo es incremental. Cada fase se explica, se implementa, se verifica y se registra en un commit antes de comenzar la siguiente.

## Ubicacion del repositorio

El codigo y los datos locales viven en el repositorio actual. El entorno Python esta separado del repositorio para no modificar la instalacion global.
