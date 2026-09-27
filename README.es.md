# Clasificador de imagenes: gatos y perros

Proyecto de aprendizaje automatico que entrena una red EfficientNet-B0 para
clasificar imagenes como `cat` o `dog`.

## Como usar y probar el proyecto

Todos los comandos siguientes se ejecutan desde la raiz del repositorio.

### 1. Instalar los requisitos

Se recomienda Python `3.11`, porque es la version utilizada para TensorFlow en
este proyecto. Crea un entorno virtual e instala las dependencias:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Linux o macOS, activa el entorno con:

```bash
source .venv/bin/activate
```

Si ya existe el entorno aislado del proyecto en `C:\venvs\calificador_imagenes`,
puedes ejecutar directamente sus scripts con:

```powershell
C:\venvs\calificador_imagenes\Scripts\python.exe src\build_model.py
```

### 2. Colocar el dataset

Descarga el dataset Dogs vs Cats y coloca las 25.000 imágenes originales en:

```text
data/raw/train/
```

Los nombres deben seguir el formato `cat.0.jpg` y `dog.0.jpg`. La estructura
inicial debe ser similar a:

```text
data/raw/train/
├── cat.0.jpg
├── cat.1.jpg
├── dog.0.jpg
└── dog.1.jpg
```

El dataset y los modelos generados están excluidos de Git mediante `.gitignore`.

### 3. Explorar y validar el dataset

Este comando comprueba los nombres, formatos, etiquetas y dimensiones de las
imágenes. También genera una muestra en
`docs/figures/dataset-samples.png`:

```powershell
python src\explore_dataset.py
```

Resultado esperado:

```text
Imagenes encontradas: 25000
Etiquetas: {'cat': 12500, 'dog': 12500}
Archivos no validos: 0
```

Para probar el script sin el dataset completo, prepara un directorio con
imágenes de prueba que respeten el formato de nombres y usa:

```powershell
python src\explore_dataset.py --dataset-dir ruta\al\dataset
```

### 4. Preparar las imágenes

Divide el dataset en 80% para entrenamiento y 20% para prueba, dentro de
`data/processed/`:

```powershell
python src\prepare_dataset.py
```

Este comando crea:

```text
data/processed/
├── train/cat/
├── train/dog/
├── test/cat/
└── test/dog/
```

También valida los generadores de Keras y muestra batches con forma
`(32, 224, 224, 3)`. Si solo quieres crear la división sin cargar TensorFlow:

```powershell
python src\prepare_dataset.py --skip-generators
```

### 5. Probar la arquitectura

Antes de entrenar, verifica que TensorFlow puede cargar EfficientNet-B0 y que el
modelo produce dos probabilidades por imagen:

```powershell
python src\build_model.py
```

Resultado esperado al final:

```text
Entrada de prueba: (2, 224, 224, 3)
Salida del modelo: (2, 2)
Verificacion del modelo completada correctamente.
```

Esta prueba no entrena el modelo.

### 6. Entrenar, optimizar y evaluar

Ejecuta el pipeline completo:

```powershell
python src\train_model.py
```

El pipeline realiza lo siguiente:

- Entrena un modelo inicial.
- Reserva el 10% del conjunto de entrenamiento para validación.
- Usa `ModelCheckpoint` para guardar la mejor versión.
- Usa `EarlyStopping` para evitar sobreajuste.
- Evalúa el mejor modelo con el conjunto de prueba.
- Genera historial, métricas y matriz de confusión.

Para una prueba rápida del flujo con pocas épocas:

```powershell
python src\train_model.py --baseline-epochs 0 --epochs 1 --patience 0
```

Para un entrenamiento más largo:

```powershell
python src\train_model.py --baseline-epochs 1 --epochs 10 --patience 3
```

El entrenamiento puede tardar bastante en CPU. Las opciones disponibles son:

```text
--data-dir          Directorio de data/processed
--models-dir        Directorio de salida de modelos
--artifacts-dir     Directorio de graficas y resultados
--batch-size        Numero de imagenes por batch; valor por defecto: 32
--baseline-epochs   Epocas del entrenamiento inicial; valor por defecto: 1
--epochs            Epocas del entrenamiento optimizado; valor por defecto: 10
--patience          Epocas sin mejora antes de detenerse; valor por defecto: 3
--seed              Semilla reproducible; valor por defecto: 42
```

Al finalizar se crean:

```text
models/cats_vs_dogs.keras       Modelo final reutilizable
models/best_model.keras         Mejor checkpoint de validación
models/metadata.json            Clases, tamaño de entrada y métricas
artifacts/history.json          Historial de entrenamiento
artifacts/training_history.png  Gráficas de pérdida y accuracy
artifacts/confusion_matrix.png  Matriz de confusión
```

### 7. Probar una predicción

Después de entrenar, clasifica una imagen individual:

```powershell
python src\app.py ruta\a\imagen.jpg
```

También puedes invocar directamente el módulo de predicción:

```powershell
python src\predict.py ruta\a\imagen.jpg
```

Para usar otro modelo:

```powershell
python src\app.py ruta\a\imagen.jpg --model ruta\al\modelo.keras
```

Resultado esperado:

```text
Prediccion: dog
Confianza: 94.32%
```

La imagen se convierte a RGB, se redimensiona a `224 x 224`, se normaliza y se
envía al modelo.

### Pruebas rápidas de código

Para comprobar que los archivos Python son válidos y que las interfaces de
terminal funcionan sin iniciar un entrenamiento:

```powershell
python -m compileall -q src
python src\build_model.py
python src\train_model.py --help
python src\predict.py --help
```

`compileall` comprueba la sintaxis. `build_model.py` requiere TensorFlow, pero
no necesita el dataset. `train_model.py --help` y `predict.py --help` solo
muestran las opciones disponibles.

## Stack tecnologico

- **Python 3.11**: ejecuta el pipeline y mantiene compatibilidad con TensorFlow.
- **TensorFlow 2.21 y Keras**: construyen, entrenan y cargan la red neuronal.
- **EfficientNet-B0**: extrae caracteristicas visuales de las imagenes.
- **Pillow**: valida, abre, convierte y redimensiona imagenes.
- **NumPy**: procesa arrays y calcula la clase con mayor probabilidad.
- **Matplotlib**: crea muestras, historial de entrenamiento y matriz de confusion.
- **scikit-learn**: calcula precision, recall, F1 y matriz de confusion.
- **Git**: versiona el codigo y excluye datasets y modelos pesados.

La explicacion detallada de cada libreria, componente y archivo esta en
[`memory_bank/technology_stack.md`](memory_bank/technology_stack.md).

## Requisitos

Se requiere Python compatible con la version de TensorFlow instalada. Instala
las dependencias con `pip install -r requirements.txt`.

---

## Estructura y referencias

La guia operativa principal esta en la seccion **Como usar y probar el
proyecto**. La informacion que aparece debajo corresponde a la plantilla
original del bootcamp y no es necesaria para entrenar ni probar este
clasificador.

Esta plantilla está diseñada para impulsar proyectos de ciencia de datos proporcionando una configuración básica para conexiones de base de datos, procesamiento de datos, y desarrollo de modelos de aprendizaje automático. Incluye una organización estructurada de carpetas para tus conjuntos de datos y un conjunto de paquetes de Python predefinidos necesarios para la mayoría de las tareas de ciencia de datos.

## Estructura

El proyecto está organizado de la siguiente manera:

- **`src/app.py`** → Script principal de Python donde correrá tu proyecto.
- **`src/explore.ipynb`** → Notebook para exploración y pruebas. Una vez finalizada la exploración, migra el código limpio a `app.py`.
- **`src/utils.py`** → Funciones auxiliares, como conexión a bases de datos.
- **`requirements.txt`** → Lista de paquetes de Python necesarios.
- **`models/`** → Contendrá tus clases de modelos SQLAlchemy.
- **`data/`** → Almacena los datasets en diferentes etapas:
  - **`data/raw/`** → Datos sin procesar.
  - **`data/interim/`** → Datos transformados temporalmente.
  - **`data/processed/`** → Datos listos para análisis.


## ⚡ Configuración Inicial en Codespaces (Recomendado)

No es necesario realizar ninguna configuración manual, ya que **Codespaces se configura automáticamente** con los archivos predefinidos que ha creado la academia para ti. Simplemente sigue estos pasos:

1. **Espera a que el entorno se configure automáticamente**.
   - Todos los paquetes necesarios y la base de datos se instalarán por sí mismos.
   - El `username` y `db_name` creados automáticamente están en el archivo **`.env`** en la raíz del proyecto.
2. **Una vez que Codespaces esté listo, puedes comenzar a trabajar inmediatamente**.


## 💻 Configuración en Local (Solo si no puedes usar Codespaces)

**Prerrequisitos**

Asegúrate de tener Python 3.11+ instalado en tu máquina. También necesitarás pip para instalar los paquetes de Python.

**Instalación**

Clona el repositorio del proyecto en tu máquina local.

Navega hasta el directorio del proyecto e instala los paquetes de Python requeridos:

```bash
pip install -r requirements.txt
```

**Crear una base de datos (si es necesario)**

Crea una nueva base de datos dentro del motor Postgres personalizando y ejecutando el siguiente comando: 

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER mi_usuario WITH PASSWORD 'mi_contraseña'; 
    CREATE DATABASE mi_base_de_datos OWNER mi_usuario; 
END \$\$;"
```
Conéctate al motor Postgres para usar tu base de datos, manipular tablas y datos: 

```bash
$ psql -U mi_usuario -d mi_base_de_datos
```

¡Una vez que estés dentro de PSQL podrás crear tablas, hacer consultas, insertar, actualizar o eliminar datos y mucho más!

**Variables de entorno**

Crea un archivo .env en el directorio raíz del proyecto para almacenar tus variables de entorno, como tu cadena de conexión a la base de datos:

```makefile
DATABASE_URL="postgresql://<USUARIO>:<CONTRASEÑA>@<HOST>:<PUERTO>/<NOMBRE_BD>"

#example
DATABASE_URL="postgresql://mi_usuario:mi_contraseña@localhost:5432/mi_base_de_datos"
```

## Ejecutando la Aplicación

Para clasificar una imagen, primero debes haber generado el modelo con
`python src/train_model.py`. Despues ejecuta:

```bash
python src/app.py ruta/a/imagen.jpg
```

## Añadiendo Modelos

Para añadir clases de modelos SQLAlchemy, crea nuevos archivos de script de Python dentro del directorio models/. Estas clases deben ser definidas de acuerdo a tu esquema de base de datos.

Definición del modelo de ejemplo (`models/example_model.py`):

```py
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Trabajando con Datos

Puedes colocar tus conjuntos de datos brutos en el directorio data/raw, conjuntos de datos intermedios en data/interim, y los conjuntos de datos procesados listos para el análisis en data/processed.

Para procesar datos, puedes modificar el script app.py para incluir tus pasos de procesamiento de datos, utilizando pandas para la manipulación y análisis de datos.

## Contribuyentes

Esta plantilla fue construida como parte del [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) de 4Geeks Academy por [Alejandro Sanchez](https://twitter.com/alesanchezr) y muchos otros contribuyentes. Descubre más sobre [los programas BootCamp de 4Geeks Academy](https://4geeksacademy.com/us/programs) aquí.

Otras plantillas y recursos como este se pueden encontrar en la página de GitHub de la escuela.
