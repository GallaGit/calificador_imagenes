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

## Implementacion

La separacion se ejecuta desde la raiz del proyecto con:

```bash
python src/prepare_dataset.py
```

El proceso utiliza la semilla `42`, por lo que produce siempre la misma separacion. La configuracion actual usa `batch_size=32` y redimensiona las imagenes a `224 x 224` al cargarlas, sin crear copias redimensionadas de cada JPG.

TensorFlow se incluye en `requirements.txt`. Si no esta instalado, primero hay que ejecutar:

```bash
pip install -r requirements.txt
```

Para crear solamente la separacion en un entorno donde TensorFlow aun no esta instalado:

```bash
python src/prepare_dataset.py --skip-generators
```

La prueba de los generadores requiere una version de Python compatible con TensorFlow. El entorno local utilizado para esta fase tiene Python `3.14.6`, version para la que no hay una instalacion de TensorFlow disponible actualmente.

## Entorno utilizado

Para no modificar la instalacion global de Python, el proyecto utiliza un entorno virtual externo en:

```text
C:\venvs\calificador_imagenes
```

Este entorno usa Python `3.11.9` y TensorFlow `2.21.0`. Para ejecutar el script con este entorno:

```bash
C:/venvs/calificador_imagenes/Scripts/python.exe src/prepare_dataset.py
```

La RTX 4060 no se utiliza desde TensorFlow instalado directamente en Windows nativo. El entrenamiento funcionara en CPU en esta configuracion. Para usar la GPU sera necesario configurar WSL2 o Docker con CUDA.

## Resultado esperado

Dos generadores funcionales: uno para entrenar y otro para evaluar. Las imagenes no se cargaran todas simultaneamente en memoria.

## Criterio para avanzar

Los generadores deben encontrar las dos clases, producir batches con forma `(batch, 224, 224, 3)` y devolver etiquetas validas.

## Resultado verificado

La separacion esperada es:

```text
train/cat: 10000
train/dog: 10000
test/cat: 2500
test/dog: 2500
```
