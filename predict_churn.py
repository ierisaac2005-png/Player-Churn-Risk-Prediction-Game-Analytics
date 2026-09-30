"""python predict_churn.py --input ejemplos_entrada.csv --model pipeline_churn.joblib"""
import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from churn_features import predict_records

def main():
    parser = argparse.ArgumentParser(description='Inferencia de churn posterior (2+ sesiones).')
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', default='pipeline_churn.joblib')
    parser.add_argument('--output', default='predicciones.csv')
    args = parser.parse_args()
    try:
        # Cargar únicamente artefactos propios o de fuentes confiables.
        bundle = joblib.load(args.model)
        if sklearn.__version__ != bundle['metadata']['sklearn_version']:
            raise ValueError('Versión de scikit-learn diferente. Instala requirements.txt del paquete o reentrena en tu entorno.')
        records = pd.read_csv(args.input)
        results, alerts = predict_records(records, bundle)
        results.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(results.to_string(index=False))
        for alert in alerts:
            print('AVISO:', alert)
        print('Archivo generado:', Path(args.output).resolve())
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print('Entrada o artefacto inválido:', exc, file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
