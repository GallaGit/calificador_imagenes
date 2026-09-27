# Fase 6: Guardar y documentar el resultado

## Objetivo

Dejar el modelo listo para reutilizarlo sin tener que entrenarlo de nuevo.

## Actividades

1. Guardar el modelo final en `models/`.
2. Registrar el tamano de entrada y el orden de las clases.
3. Registrar las metricas obtenidas.
4. Documentar como cargar el modelo.
5. Mantener fuera de Git los datasets y archivos pesados.

## Resultado esperado

Un archivo de modelo reutilizable, una explicacion de sus resultados y unas instrucciones claras para reproducir el proyecto.

## Usar el modelo

```powershell
python src\app.py ruta\a\imagen.jpg
```

El modelo queda en `models/cats_vs_dogs.keras`, los metadatos y metricas en
`models/metadata.json`, y las graficas y matriz de confusion en `artifacts/`.
