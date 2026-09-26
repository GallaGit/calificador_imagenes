# Fase 2: Preparar las imagenes

## Objetivo

Convertir el conjunto de imagenes en una entrada que Keras pueda leer de forma eficiente durante el entrenamiento.

## Explicacion sencilla

Las fotografias originales tienen diferentes formas y tamanos. La red neuronal necesita recibir imagenes con el mismo formato. Por eso las cargaremos en batches y las redimensionaremos a `224 x 224` pixeles con tres canales de color RGB.

Usaremos `ImageDataGenerator` y `flow_from_directory()`. Este metodo lee las imagenes progresivamente y no intenta cargar todo el dataset en la memoria.

## Actividades

1. Separar las imagenes en entrenamiento y prueba.
2. Crear las carpetas `cat` y `dog` dentro de cada conjunto.
3. Configurar el tamano `224 x 224`.
4. Configurar el tamano del batch.
5. Crear los generadores de entrenamiento y prueba.
6. Comprobar que Keras asigna correctamente las clases.

## Resultado esperado

Dos generadores funcionales: uno para entrenar y otro para evaluar. Las imagenes no se cargaran todas simultaneamente en memoria.

## Criterio para avanzar

Los generadores deben encontrar las dos clases, producir batches con forma `(batch, 224, 224, 3)` y devolver etiquetas validas.
