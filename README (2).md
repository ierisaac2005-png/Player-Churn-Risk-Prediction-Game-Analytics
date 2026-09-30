# Player Churn Risk Prediction — Game Analytics

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ierisaac2005-png/Player-Churn-Risk-Prediction-Game-Analytics/blob/main/Player_Churn_Risk_Prediction.ipynb)

Proyecto de clasificación binaria para analizar el abandono de jugadores a partir de datos históricos de actividad y telemetría. Integra exploración y preparación de datos, comparación de modelos, validación cruzada, optimización, interpretabilidad e inferencia con un modelo guardado.

**Autores:** Isaac Esquivel Ruiz y Gael Rodríguez Jiménez.

**Resultado final:** Random Forest ajustado, con umbral de decisión de **0.39**, detectó **82 de 99 abandonos** en el conjunto de prueba y generó **11 falsas alertas**. Los datos son sintéticos; este resultado demuestra desempeño experimental, no un efecto comprobado sobre la retención de jugadores reales.

## Objetivo

Estimar el riesgo de churn posterior entre jugadores que regresaron al juego al menos una vez. El abandono inmediato se analiza como un fenómeno distinto para evitar que los jugadores de una sola sesión dominen artificialmente el entrenamiento.

La utilidad propuesta es ayudar a un equipo de retención a priorizar jugadores para revisión humana. Una alerta identifica riesgo; no demuestra que una intervención evitará el abandono.

## Hallazgos principales

- Dataset original: **2,096 jugadores y 187 variables**.
- Abandono inmediato: **273 jugadores** registraron una sola sesión; **97.80%** presentan churn.
- Población de modelado: **1,823 jugadores** con dos o más sesiones.
- Tasa de churn en la población modelada: **35.93%**.
- División estratificada: **1,276** registros de entrenamiento, **273** de validación y **274** de prueba; aproximadamente 70%, 15% y 15%.
- Semilla utilizada: **42**.

### Evolución desde la fase inicial

La fase 1 estableció referencias con DummyClassifier, Regresión Logística y Árbol de Decisión. El árbol obtuvo F1 de **0.7200** en validación; la regresión logística alcanzó ROC-AUC de **0.8031** y recall de **0.6939** al utilizar un umbral de 0.40.

La fase 2 amplió la comparación con Random Forest, Gradient Boosting y SVM, incorporó ingeniería y selección de características dentro del pipeline, validación cruzada y búsquedas de hiperparámetros. Las métricas iniciales corresponden a validación y no deben confundirse con la evaluación final en prueba.

### Resultados finales en prueba

El modelo y su umbral se seleccionaron con entrenamiento y validación **antes de evaluar el conjunto de prueba**.

| Modelo | Umbral | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest ajustado — seleccionado** | **0.39** | **0.8978** | **0.8817** | **0.8283** | **0.8542** | **0.9255** |
| SVM ajustado — finalista | 0.64 | 0.8796 | 0.9125 | 0.7374 | 0.8156 | 0.8892 |

**Matriz de confusión del modelo seleccionado:**

| Estado real | Predice permanencia | Predice abandono |
|---|---:|---:|
| Permanece | 164 | 11 |
| Abandona | 17 | 82 |

En términos operativos, detectó **82.83% de los abandonos** y **88.17% de sus alertas fueron correctas**. Frente al otro finalista, detectó nueve abandonos adicionales a cambio de cuatro falsas alertas más. Esta comparación describe los resultados; no se utilizó para volver a seleccionar el ganador con datos de prueba.

## Metodología

1. Comprensión y auditoría del dataset: faltantes, duplicados, constantes y valores extremos.
2. Separación entre abandono inmediato y churn posterior.
3. Revisión de predictores con riesgo de fuga de información o baja capacidad de generalización.
4. División estratificada en entrenamiento, validación y prueba.
5. Ingeniería de dos razones de actividad: sesiones recientes respecto a la semana y stages por sesión semanal.
6. Preprocesamiento con `Pipeline` y `ColumnTransformer`: imputación, escalamiento, codificación, eliminación de constantes y selección de características cuando corresponde.
7. Comparación de modelos mediante cinco folds estratificados sobre entrenamiento, con F1 como métrica principal.
8. Optimización de hiperparámetros y comparación de desempeño, estabilidad, interpretabilidad y costo computacional.
9. Ajuste de umbrales en validación, buscando recall de al menos 0.70 y priorizando F1 entre las opciones factibles.
10. Evaluación final en prueba, análisis de falsos positivos y negativos, curvas ROC y precision-recall.
11. Interpretabilidad global por permutación, sensibilidad local y comparación por grupos de actividad.
12. Exportación del pipeline, validación de entradas, prueba de inferencia en un proceso separado y simulación de monitoreo.

Las transformaciones se ajustan dentro de cada fold durante la validación cruzada. El conjunto de prueba se utiliza para la evaluación final, no para entrenar ni ajustar el umbral.

## Estructura del repositorio

| Archivo | Función |
|---|---|
| [Player_Churn_Risk_Prediction.ipynb](Player_Churn_Risk_Prediction.ipynb) | Desarrollo del proyecto, gráficas, experimentos y resultados ejecutados. |
| [player-churn.csv](player-churn.csv) | Dataset original utilizado en el análisis. |
| [pipeline_churn.joblib](pipeline_churn.joblib) | Pipeline entrenado, umbral y metadatos para inferencia. |
| [churn_features.py](churn_features.py) | Transformaciones, validación del esquema y predicción compartidas. |
| [predict_churn.py](predict_churn.py) | Ejecución de predicciones desde la terminal. |
| [ejemplos_entrada.csv](ejemplos_entrada.csv) | Cinco registros de ejemplo con el esquema de entrada esperado. |
| [requirements.txt](requirements.txt) | Versiones de dependencias para el modelo publicado. |
| [README.md](README.md) | Descripción, resultados e instrucciones de uso. |

## Ejecución

### Proyecto completo en Google Colab

1. Abre el notebook con el botón **Open in Colab**.
2. Ejecuta todas las celdas en orden.
3. Cuando se solicite, carga únicamente `player-churn.csv`. Si ya está disponible en el directorio de ejecución, la celda de carga lo detecta.
4. Al finalizar, revisa los resultados y los archivos generados en `churn_artifacts`.

El notebook genera el modelo y un `requirements.txt` con las versiones del entorno donde se entrenó. Conserva ambos juntos si utilizas un modelo regenerado en Colab: sus versiones pueden diferir de las del modelo publicado en este repositorio.

### Usar el modelo publicado sin reentrenarlo

Clona o descarga el repositorio y abre una terminal en su carpeta. Utiliza Python 3.11 o posterior y, preferentemente, un entorno virtual.

```bash
python -m pip install -r requirements.txt
python predict_churn.py --input ejemplos_entrada.csv --model pipeline_churn.joblib --output predicciones.csv
```

El comando imprime los resultados y crea `predicciones.csv` con:

- `probabilidad_churn`: puntuación de probabilidad estimada por el modelo.
- `prediccion_churn`: 1 indica riesgo de abandono y 0 permanencia prevista.
- `umbral`: punto de decisión aplicado, 0.39 en el modelo publicado.
- `version_modelo` y `aviso`: identificación del modelo y limitación de uso.

La clase positiva se asigna cuando la probabilidad estimada es mayor o igual al umbral. Estas probabilidades no cuentan con una calibración verificada para jugadores reales.

### Requisitos de entrada

Utiliza `ejemplos_entrada.csv` como referencia del esquema. El dataset original completo no es directamente una entrada de inferencia: incluye identificadores, la etiqueta y otras columnas fuera del esquema esperado.

- Se requieren exactamente las columnas indicadas por los metadatos del modelo, con valores numéricos.
- Los jugadores deben tener al menos dos sesiones.
- Las categorías deben ser válidas; los conteos deben ser enteros no negativos.
- Se rechazan columnas faltantes o adicionales, valores infinitos y telemetría negativa.
- Los nulos imputables generan advertencias; los campos de elegibilidad y categoría requieren valores válidos.
- Los valores fuera del rango observado en entrenamiento generan advertencias para revisión.

El script verifica la versión de scikit-learn antes de predecir. Para el artefacto publicado, instala el `requirements.txt` del repositorio. Carga archivos `.joblib` únicamente de fuentes confiables.

### Ejecución local del notebook

Además de las dependencias del modelo, instala las herramientas de notebook y visualización:

```bash
python -m pip install jupyter matplotlib seaborn
jupyter notebook Player_Churn_Risk_Prediction.ipynb
```

Mantén `player-churn.csv` en el directorio de ejecución. La celda de carga lo detecta sin necesitar la carga interactiva de Colab.

## Alcance y limitaciones

- **Datos sintéticos:** las métricas no demuestran generalización a un videojuego real.
- **Población específica:** el modelo aplica a jugadores con dos o más sesiones; no predice abandono inmediato.
- **Validez temporal pendiente:** las variables deben estar disponibles antes de la predicción y del abandono. La partición aleatoria estratificada no sustituye una evaluación temporal prospectiva.
- **Impacto no medido:** no se ha demostrado ahorro económico, mejora causal de retención ni retorno de inversión.
- **Interpretabilidad limitada:** la sensibilidad local describe cambios del modelo al modificar entradas; no demuestra causalidad.
- **Revisión humana:** una alerta es apoyo para priorizar casos, no una instrucción automática para conceder incentivos o tomar decisiones sobre jugadores.

## Siguiente fase

La fase académica de comparación, optimización, evaluación final e inferencia está implementada. La evolución como proyecto personal contempla:

1. Validar con telemetría real y periodos futuros, definiendo claramente las ventanas de observación y abandono.
2. Evaluar calibración, costos de los errores y capacidad de atención del equipo.
3. Realizar un piloto supervisado que mida el efecto de las acciones de retención.
4. Aplicar monitoreo de calidad de datos, cambios de distribución y desempeño con nuevas etiquetas.

El notebook incluye una simulación de monitoreo; todavía no constituye un servicio desplegado ni una operación validada en producción.
