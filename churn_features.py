"""Transformaciones y validación compartidas por entrenamiento e inferencia."""
import numpy as np
import pandas as pd

def engineer_features(frame):
    out = frame.copy()
    week = out['begin_session_count_last_week(-1)']
    day = out['begin_session_count_last_day(-1)']
    stages = out['begin_stage_count_last_week(-1)']
    # Un denominador cero se representa como ausencia de actividad, no infinito.
    out['recent_session_share'] = day.div(week.replace(0, np.nan)).fillna(0)
    out['stages_per_session_week'] = stages.div(week.replace(0, np.nan)).fillna(0)
    return out

def validate_input(frame, metadata):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Se requiere una tabla no vacía.')
    if frame.columns.duplicated().any():
        raise ValueError('Hay nombres de columnas duplicados.')
    expected = metadata['input_columns']
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f'Columnas faltantes: {missing}; columnas adicionales: {extra}')
    clean = frame.loc[:, expected].copy()
    if any(not pd.api.types.is_numeric_dtype(clean[c]) for c in expected):
        raise ValueError('Todas las columnas deben contener valores numéricos.')
    if np.isinf(clean.to_numpy(dtype=float)).any():
        raise ValueError('La entrada contiene infinito.')
    # NaN en telemetría se imputa; elegibilidad y categoría requieren valor explícito.
    if clean['session_count'].isna().any() or (clean['session_count'] < 2).any():
        raise ValueError('El modelo solo aplica a jugadores con al menos dos sesiones.')
    if not clean['cohort_day_of_week'].isin(metadata['allowed_days']).all():
        raise ValueError('Categoría de cohort_day_of_week no reconocida.')
    count_cols = [c for c in expected if 'count' in c]
    for c in count_cols:
        values = clean[c].dropna()
        if (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f'{c}: se requieren conteos enteros no negativos.')
    if (clean.drop(columns=['cohort_day_of_week']) < 0).any().any():
        raise ValueError('La telemetría no puede contener valores negativos.')
    alerts = []
    if clean.isna().any().any():
        alerts.append('Hay valores nulos: se aplicará la imputación aprendida.')
    extreme = []
    for c, limits in metadata['training_ranges'].items():
        if ((clean[c] < limits[0]) | (clean[c] > limits[1])).any():
            extreme.append(c)
    if extreme:
        alerts.append('Valores fuera del rango observado en entrenamiento: ' + ', '.join(extreme))
    return clean, alerts

def predict_records(frame, bundle):
    clean, alerts = validate_input(frame, bundle['metadata'])
    model = bundle['pipeline']
    positive = list(model.classes_).index(1)
    probability = model.predict_proba(clean)[:, positive]
    threshold = bundle['threshold']
    result = pd.DataFrame({
        'probabilidad_churn': probability,
        'prediccion_churn': (probability >= threshold).astype(int),
        'umbral': threshold,
        'version_modelo': bundle['metadata']['version'],
        'aviso': 'Estimación sobre datos sintéticos; requiere revisión humana.'
    }, index=frame.index)
    return result, alerts
