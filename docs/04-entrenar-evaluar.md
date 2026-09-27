# Fase 4: Entrenar y evaluar el modelo

## Objetivo

Enseñar al modelo a distinguir perros y gatos y medir su rendimiento con imagenes que no utilizo para aprender.

## Explicacion sencilla

Durante el entrenamiento, el modelo hace predicciones, compara sus respuestas con las etiquetas correctas y ajusta sus parametros. El conjunto de prueba sirve para comprobar si aprendio patrones generales y no solo memorizo las imagenes de entrenamiento.

## Actividades

1. Entrenar el modelo durante un numero inicial de epocas.
2. Guardar las metricas de perdida y accuracy.
3. Graficar la evolucion del entrenamiento.
4. Evaluar con el conjunto de prueba.
5. Analizar errores y aciertos.

## Resultado esperado

Una primera medida objetiva del rendimiento y graficas que permitan detectar sobreajuste.

## Ejecucion

Desde la raiz del proyecto:

```powershell
python src\train_model.py --baseline-epochs 1 --epochs 10
```

El script reserva el 10% del conjunto de entrenamiento para validacion y mantiene el
conjunto de prueba separado hasta la evaluacion final. Guarda el historial en
`artifacts/history.json` y la grafica en `artifacts/training_history.png`.
