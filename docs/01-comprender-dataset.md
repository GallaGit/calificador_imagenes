# Fase 1: Comprender y explorar el dataset

## Objetivo

Conocer como estan organizadas las imagenes antes de entrenar el modelo. En esta fase comprobaremos que los archivos existen, que sus nombres contienen la etiqueta correcta y que hay ejemplos de perros y gatos.

## Explicacion sencilla

Cada fotografia es un dato. El nombre de la fotografia nos dice la respuesta esperada:

- `dog` representa un perro.
- `cat` representa un gato.

La respuesta esperada se llama etiqueta. El modelo aprendera relacionando los pixeles de cada fotografia con esa etiqueta.

## Implementacion realizada

La fase esta implementada en `src/explore_dataset.py`. El script busca imagenes
`.jpg`, `.jpeg` y `.png`, reconoce nombres como `cat.123.jpg` y `dog.456.jpg`,
valida cada archivo con Pillow, cuenta las clases y dimensiones, informa
archivos invalidos y genera una cuadricula con nueve gatos y nueve perros.

Tambien falla deliberadamente si el dataset no contiene las cantidades
esperadas, para evitar avanzar con datos incompletos.

## Actividades

1. Localizar la carpeta original del dataset.
2. Contar las imagenes de perros y gatos.
3. Revisar algunos nombres de archivo.
4. Mostrar las primeras nueve imagenes de perros.
5. Mostrar las primeras nueve imagenes de gatos.
6. Revisar que las imagenes sean a color y tengan tamanos variados.

## Ejecucion y parametros

Desde la raiz del proyecto:

```powershell
python src\explore_dataset.py
```

Se pueden cambiar la ruta y la salida de la figura:

```powershell
python src\explore_dataset.py --dataset-dir ruta\al\dataset --figure-path ruta\a\muestras.png
```

## Resultado esperado

Un informe en consola y una figura que confirman que el dataset se puede leer y
contiene las dos clases esperadas.

El script genera la figura `docs/figures/dataset-samples.png`. La figura se ignora en Git porque es un artefacto generado; puede abrirse localmente para revisar las muestras.

Con el dataset completo se espera `25000` imagenes, `12500` por clase y cero
archivos invalidos.

## Criterio para avanzar

No debe haber archivos ilegibles ni nombres de imagen desconocidos. Deben
existir las clases `cat` y `dog`, y deben quedar confirmadas la ruta
`data/raw/train`, la entrada `224 x 224` y la division 80/20.

## Decisiones confirmadas

- Dataset local: `data/raw/train/`.
- Entrada del modelo: `224 x 224 x 3`.
- Division prevista: 80% entrenamiento y 20% prueba.
- Hardware disponible: NVIDIA GeForce RTX 4060.

## Conexion con la siguiente fase

La fase 1 no modifica las imagenes. Sus resultados sirven como entrada para
`src/prepare_dataset.py`, que crea la estructura procesada para Keras.
