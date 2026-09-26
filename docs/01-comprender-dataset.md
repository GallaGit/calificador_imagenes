# Fase 1: Comprender y explorar el dataset

## Objetivo

Conocer como estan organizadas las imagenes antes de entrenar el modelo. En esta fase comprobaremos que los archivos existen, que sus nombres contienen la etiqueta correcta y que hay ejemplos de perros y gatos.

## Explicacion sencilla

Cada fotografia es un dato. El nombre de la fotografia nos dice la respuesta esperada:

- `dog` representa un perro.
- `cat` representa un gato.

La respuesta esperada se llama etiqueta. El modelo aprendera relacionando los pixeles de cada fotografia con esa etiqueta.

## Actividades

1. Localizar la carpeta original del dataset.
2. Contar las imagenes de perros y gatos.
3. Revisar algunos nombres de archivo.
4. Mostrar las primeras nueve imagenes de perros.
5. Mostrar las primeras nueve imagenes de gatos.
6. Revisar que las imagenes sean a color y tengan tamanos variados.

## Resultado esperado

Un informe o notebook de exploracion que confirme que el dataset se puede leer y que contiene las dos clases esperadas.

La exploracion se ejecuta desde la raiz del proyecto con:

```bash
python src/explore_dataset.py
```

El script genera la figura `docs/figures/dataset-samples.png`. La figura se ignora en Git porque es un artefacto generado; puede abrirse localmente para revisar las muestras.

## Criterio para avanzar

No debe haber archivos ilegibles o etiquetas desconocidas en la muestra revisada. La ruta del dataset y las decisiones de tamano y division deben estar confirmadas.

## Decisiones confirmadas

- Dataset local: `data/raw/train/`.
- Entrada del modelo: `224 x 224 x 3`.
- Division prevista: 80% entrenamiento y 20% prueba.
- Hardware disponible: NVIDIA GeForce RTX 4060.
