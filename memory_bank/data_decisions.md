# Decisiones sobre los datos

## Dataset original

El dataset se copio localmente desde la carpeta descargada a:

```text
data/raw/train/
```

Contiene 25.000 imagenes JPG:

- 12.500 gatos.
- 12.500 perros.

El dataset, sus imagenes procesadas y las figuras generadas estan excluidos de Git mediante `.gitignore`.

## Validacion realizada

El script `src/explore_dataset.py` comprobo:

- 25.000 imagenes encontradas.
- 12.500 archivos de cada clase.
- 8.513 tamanos de imagen diferentes.
- Tamano mas frecuente: `500 x 374`.
- Cero archivos invalidos o corruptos.

La figura de muestras se genera en `docs/figures/dataset-samples.png` y no se versiona.

## Separacion

El script `src/prepare_dataset.py` crea una division reproducible con semilla `42`:

```text
data/processed/
├── train/
│   ├── cat/  10000 imagenes
│   └── dog/  10000 imagenes
└── test/
    ├── cat/   2500 imagenes
    └── dog/   2500 imagenes
```

La proporcion es 80% para entrenamiento y 20% para prueba, manteniendo el equilibrio entre clases.

## Generadores

La configuracion validada es:

- `target_size=(224, 224)`.
- `batch_size=32`.
- `class_mode="categorical"`.
- `rescale=1/255`.
- `cat` se asigna a `0`.
- `dog` se asigna a `1`.

Resultado de un batch validado:

```text
Imagenes: (32, 224, 224, 3)
Etiquetas: (32, 2)
```

Las imagenes no se redimensionan fisicamente en disco; Keras las redimensiona al cargarlas. Esto evita duplicar innecesariamente el dataset.
