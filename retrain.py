import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from api.utilities.s3_storage import S3ModelStorage

# --- Logger -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuracion ------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

COLS_NUMERICAS = [
    "edad",
    "tamano_tumor_mm",
    "marcador_ca125",
    "marcador_cea",
    "densidad_celular",
]
COLS_CATEGORICAS = ["forma_tumor", "textura"]
COLS_BINARIAS = ["historial_familiar", "fumador"]
TARGET = "diagnostico"
COL_DROP = ["paciente_id"]

PARAM_GRID = {
    "clasificador__n_estimators": [50, 100, 200],
    "clasificador__max_depth": [5, 10, 20],
    "clasificador__min_samples_split": [2, 5, 10],
}


# --- Funciones ----------------------------------------------------------------
def load_data(data_path):
    df = pd.read_csv(data_path)
    expected = set(COLS_NUMERICAS + COLS_CATEGORICAS + COLS_BINARIAS + COL_DROP + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el dataset: {missing}")
    return df


def prepare_data(df):
    X = df.drop(columns=COL_DROP + [TARGET])
    y = df[TARGET]
    return X, y


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLS_NUMERICAS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), COLS_CATEGORICAS),
            ("bin", "passthrough", COLS_BINARIAS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocesador", preprocessor),
            ("clasificador", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )


def cross_validate(pipeline, X, y):
    results = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cross_val_score(pipeline, X, y, cv=CV_FOLDS, scoring=metric)
        results[f"cv_{metric}_mean"] = round(float(scores.mean()), 4)
        results[f"cv_{metric}_std"] = round(float(scores.std()), 4)
    return results


def tune_hyperparameters(pipeline, X, y):
    grid = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X, y)
    return grid


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Benigno", "Maligno"]
    )

    return metrics, cm, report


def write_github_summary(metrics, cv_metrics, best_params, cm, report):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        "## Reentrenamiento - Modelo Cancer de Mama",
        "",
        "### Mejores Hiperparametros",
        "| Parametro | Valor |",
        "|-----------|-------|",
    ]
    for k, v in best_params.items():
        param_name = k.replace("clasificador__", "")
        lines.append(f"| `{param_name}` | {v} |")

    lines += [
        "",
        "### Metricas en Test Set",
        "| Metrica | Valor |",
        "|---------|-------|",
    ]
    for k, v in metrics.items():
        lines.append(f"| **{k.upper()}** | {v:.4f} |")

    lines += [
        "",
        "### Cross-Validation (5-Fold)",
        "| Metrica | Media | Std |",
        "|---------|-------|-----|",
    ]
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        mean = cv_metrics.get(f"cv_{k}_mean", "-")
        std = cv_metrics.get(f"cv_{k}_std", "-")
        lines.append(f"| {k.upper()} | {mean} | {std} |")

    lines += [
        "",
        "### Matriz de Confusion",
        "| | Pred. Benigno | Pred. Maligno |",
        "|---|---|---|",
        f"| **Real Benigno** | {cm[0][0]} | {cm[0][1]} |",
        f"| **Real Maligno** | {cm[1][0]} | {cm[1][1]} |",
        "",
        "### Reporte de Clasificacion",
        "```",
        report,
        "```",
    ]

    with open(summary_path, "a") as f:
        f.write("\n".join(lines))


def set_github_output(metrics):
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a") as f:
        for k, v in metrics.items():
            f.write(f"{k}={v}\n")


# --- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reentrenamiento del modelo de cancer de mama")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="cancer-model")
    parser.add_argument("--data-key", required=True, help="S3 key del dataset, ej: data/dataset_cancer.csv")
    parser.add_argument("--min-accuracy", type=float, default=0.60)
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    # --------------------------------------------------------------------------
    # 0. Conexion a S3
    # --------------------------------------------------------------------------
    storage = S3ModelStorage(bucket=args.bucket, prefix=args.prefix)

    # --------------------------------------------------------------------------
    # 1. Descargar datos desde S3
    # --------------------------------------------------------------------------
    local_csv = storage.download_dataset(
        model_name=args.data_key,
        local_path="/tmp/dataset_cancer.csv"
    )
    df = load_data(local_csv)

    # --------------------------------------------------------------------------
    # 2. Preparacion de datos
    # --------------------------------------------------------------------------
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d | Test: %d", X_train.shape[0], X_test.shape[0])

    # --------------------------------------------------------------------------
    # 3. Cross-validation baseline
    # --------------------------------------------------------------------------
    pipeline = build_pipeline()
    cv_metrics = cross_validate(pipeline, X_train, y_train)
    logger.info(
        "CV accuracy: %.4f (+/- %.4f)",
        cv_metrics["cv_accuracy_mean"],
        cv_metrics["cv_accuracy_std"]
    )

    # --------------------------------------------------------------------------
    # 4. Optimizacion de hiperparametros
    # --------------------------------------------------------------------------
    grid = tune_hyperparameters(build_pipeline(), X_train, y_train)
    best_params = grid.best_params_
    logger.info("Mejor accuracy CV: %.4f", grid.best_score_)

    # --------------------------------------------------------------------------
    # 5. Evaluacion final
    # --------------------------------------------------------------------------
    best_model = grid.best_estimator_
    metrics, cm, report = evaluate_model(best_model, X_test, y_test)
    logger.info(
        "Test — accuracy: %.4f | precision: %.4f | recall: %.4f | f1: %.4f | roc_auc: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )

    # --------------------------------------------------------------------------
    # 6. Quality gate
    # --------------------------------------------------------------------------
    if metrics["accuracy"] < args.min_accuracy:
        logger.error(
            "accuracy (%.4f) por debajo del umbral (%.2f)",
            metrics["accuracy"], args.min_accuracy
        )
        write_github_summary(metrics, cv_metrics, best_params, cm, report)
        sys.exit(1)

    # --------------------------------------------------------------------------
    # 7. Subir modelo y metricas a S3
    # --------------------------------------------------------------------------
    all_metrics = {
        **metrics,
        **cv_metrics,
        "best_params": best_params,
        "confusion_matrix": cm
    }
    storage.save_model(model=best_model, metrics=all_metrics)

    # --------------------------------------------------------------------------
    # 8. Escribir resumen en GitHub Actions
    # --------------------------------------------------------------------------
    write_github_summary(metrics, cv_metrics, best_params, cm, report)
    set_github_output(metrics)

    logger.info(f"Reentrenamiento completado — modelo subido a s3://{args.bucket}/%{args.prefix}/")


if __name__ == "__main__":
    main()
