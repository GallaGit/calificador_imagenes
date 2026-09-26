# Fase 3: Construir la RNA

## Objetivo

Crear una red neuronal convolucional que aprenda caracteristicas visuales y clasifique cada imagen como gato o perro.

## Arquitectura

La red usara EfficientNet-B0 como extractor de caracteristicas, sin su clasificador final:

1. EfficientNet-B0 con entrada `224 x 224 x 3`.
2. `GlobalAveragePooling2D` para resumir los mapas de caracteristicas.
3. Capa densa de 128 neuronas con activacion ReLU.
4. Capa final de 2 neuronas con activacion softmax.

## Explicacion sencilla

EfficientNet-B0 detecta patrones visuales como bordes, formas, texturas y partes de animales. Las capas densas convierten esos patrones en una decision final entre las dos clases.

## Resultado esperado

Un modelo compilado con una funcion de perdida adecuada para dos clases y una metrica de accuracy.
