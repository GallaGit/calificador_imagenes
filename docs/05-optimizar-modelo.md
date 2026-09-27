# Fase 5: Optimizar el modelo

## Objetivo

Conservar la mejor version del modelo y detener el entrenamiento cuando deje de mejorar.

## Herramientas

- `ModelCheckpoint`: guarda automaticamente el modelo con mejores resultados.
- `EarlyStopping`: detiene el entrenamiento si la metrica de validacion no mejora durante varias epocas.

## Explicacion sencilla

No siempre el ultimo modelo entrenado es el mejor. A veces el modelo empieza a memorizar los datos de entrenamiento. Los callbacks permiten conservar el punto de entrenamiento mas util y evitar trabajo innecesario.

## Actividades

1. Configurar una ruta para el mejor modelo.
2. Anadir `ModelCheckpoint` al entrenamiento.
3. Anadir `EarlyStopping` con restauracion de los mejores pesos.
4. Cargar el mejor modelo guardado.
5. Hacer predicciones sobre el conjunto de prueba.
6. Comparar el resultado con el modelo inicial.

## Resultado esperado

Un modelo optimizado y una comparacion clara entre el primer entrenamiento y el mejor resultado.

La implementacion se encuentra en `src/train_model.py`. El mejor checkpoint se guarda
en `models/best_model.keras` y se copia como `models/cats_vs_dogs.keras` despues de
evaluarlo.
