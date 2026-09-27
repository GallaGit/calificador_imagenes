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

## Paso 1: crear el modelo

La arquitectura se implementa en `src/build_model.py` y sigue la estructura indicada en el enunciado:

```text
Entrada: 224 x 224 x 3
    -> EfficientNet-B0 sin clasificador original
    -> GlobalAveragePooling2D
    -> Dense(128, activation="relu")
    -> Dense(2, activation="softmax")
```

`weights=None` indica que EfficientNet comienza con pesos aleatorios y aprendera utilizando este dataset. `include_top=False` elimina el clasificador original de ImageNet, que tenia 1.000 clases y no sirve directamente para nuestro problema de dos clases.

El modelo se compila con Adam, `categorical_crossentropy` y `accuracy`. La funcion de perdida es compatible con las etiquetas one-hot producidas por los generadores configurados con `class_mode="categorical"`.

Para crear y verificar el modelo:

```powershell
python src\build_model.py
```

La verificacion no entrena el modelo. Solo comprueba que un batch artificial con forma `(2, 224, 224, 3)` produce una salida con forma `(2, 2)`.

## Criterio del paso 1

El modelo debe compilarse, mostrar su resumen y producir dos probabilidades por cada imagen: una para `cat` y otra para `dog`.
