Player Churn Risk Prediction — Game Analytics



Proyecto de clasificación binaria para analizar el abandono de jugadores a partir de datos históricos de actividad y telemetría. La fase 1 cubre exploración, calidad de datos, selección de predictores, preprocesamiento reproducible y comparación inicial de modelos.

Objetivo

Estimar el riesgo de churn posterior entre jugadores que regresaron al juego al menos una vez. El abandono inmediato se analiza como un fenómeno distinto para evitar que los jugadores de una sola sesión dominen artificialmente el entrenamiento.

Hallazgos principales

Dataset original: 2,096 jugadores y 187 variables.

Abandono inmediato: 273 jugadores registraron una sola sesión; 97.80% presentan churn.

Población de modelado: 1,823 jugadores con dos o más sesiones.

Tasa de churn en la población modelada: 35.93%.

Predictores finales: 159.

División estratificada: 70% entrenamiento, 15% validación y 15% prueba.

El Árbol de Decisión presentó el mejor desempeño equilibrado en validación:

Accuracy

Precision

Recall

F1

ROC-AUC

0.8205

0.8182

0.6429

0.7200

0.7764

La Regresión Logística obtuvo un ROC-AUC de 0.8031. Con un threshold de 0.40 alcanzó un recall de 0.6939, por lo que permanece como alternativa cuando reducir falsos negativos sea la prioridad operativa.

Metodología

Comprensión y auditoría del dataset.

Revisión de faltantes, duplicados, variables constantes y valores extremos.

Separación entre abandono inmediato y churn posterior.

Evaluación de variables con riesgo de leakage o baja capacidad de generalización.

Construcción reproducible de X e y.

División estratificada en entrenamiento, validación y prueba.

Imputación, escalamiento y codificación mediante Pipeline y ColumnTransformer.

Comparación de DummyClassifier, Regresión Logística y Árbol de Decisión.

Evaluación mediante accuracy, precision, recall, F1, ROC-AUC y matrices de confusión.

Análisis del threshold de clasificación.

Estructura del repositorio

.
├── Player_Churn_Risk_Prediction.ipynb
├── player-churn.csv
├── README.md
├── requirements.txt
└── .gitignore

Ejecución

El notebook está preparado para Google Colab:

Abre el notebook mediante el botón Open in Colab.

Ejecuta todas las celdas en orden.

Cuando se solicite, carga player-churn.csv.

Para una ejecución local, instala las dependencias:

python -m pip install -r requirements.txt

Después sustituye la celda de carga de Google Colab por:

df = pd.read_csv("player-churn.csv")

Alcance y limitaciones

Los datos son sintéticos y los resultados no deben generalizarse directamente a un videojuego real.

El modelo se dirige a jugadores con dos o más sesiones; no predice abandono inmediato.

Las variables históricas deben calcularse antes del momento de inferencia para evitar fuga temporal.

El conjunto de prueba permanece reservado para una evaluación posterior.

La comparación actual utiliza una sola partición de validación y no incluye optimización extensa de hiperparámetros.

Siguiente fase

La fase 2 incorporará validación cruzada, selección de características, modelos adicionales, optimización de hiperparámetros, interpretabilidad y evaluación final sobre el conjunto de prueba.
