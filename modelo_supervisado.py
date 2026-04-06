"""
modelo_supervisado.py
=====================
Modelo de aprendizaje supervisado para predecir el nivel de congestion
en las estaciones del Metro de Medellin.

Algoritmos evaluados:
  1. Arbol de Decision  (Decision Tree  - modelo principal)
  2. Bosque Aleatorio   (Random Forest  - comparacion)

Flujo del programa:
  1. Cargar y explorar el dataset
  2. Preprocesar los datos
  3. Dividir en entrenamiento / prueba
  4. Entrenar Arbol de Decision con busqueda de hiperparametros
  5. Evaluar: accuracy, reporte de clasificacion, matriz de confusion
  6. Importancia de variables
  7. Comparacion con Random Forest
  8. Guardar visualizaciones

Uso:
  python modelo_supervisado.py

Dependencias:
  pip install pandas scikit-learn matplotlib seaborn
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# ---------------------------------------------------------------------------
# Configuracion general
# ---------------------------------------------------------------------------

DATA_FILE    = "datos_transporte.csv"
OUTPUT_DIR   = "resultados"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
TARGET_COL   = "nivel_congestion"
CLASS_ORDER  = ["bajo", "medio", "alto"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Carga y exploracion de datos
# ---------------------------------------------------------------------------

def load_and_explore(path: str) -> pd.DataFrame:
    print("=" * 60)
    print("1. CARGA Y EXPLORACION DEL DATASET")
    print("=" * 60)

    df = pd.read_csv(path)

    print(f"\nDimensiones : {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"\nColumnas    : {list(df.columns)}")
    print(f"\nTipos de dato:\n{df.dtypes}")
    print(f"\nValores nulos:\n{df.isnull().sum()}")
    print(f"\nEstadisticas descriptivas:\n{df.describe()}")
    print(f"\nDistribucion del target:\n{df[TARGET_COL].value_counts()}")

    # Grafico de distribucion del target
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df[TARGET_COL].value_counts().reindex(CLASS_ORDER).plot.bar(
        ax=axes[0], color=["#4CAF50", "#FFC107", "#F44336"], edgecolor="black"
    )
    axes[0].set_title("Distribucion del nivel de congestion")
    axes[0].set_xlabel("Nivel")
    axes[0].set_ylabel("Cantidad de registros")
    axes[0].tick_params(axis="x", rotation=0)

    congestion_hora = df.groupby("hora")[TARGET_COL].value_counts(normalize=True).unstack()
    congestion_hora[CLASS_ORDER].plot.area(
        ax=axes[1],
        color=["#4CAF50", "#FFC107", "#F44336"],
        alpha=0.7,
    )
    axes[1].set_title("Congestion por hora del dia")
    axes[1].set_xlabel("Hora")
    axes[1].set_ylabel("Proporcion")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "1_exploracion.png"), dpi=150)
    plt.close()
    print(f"\n  -> Grafico guardado: {OUTPUT_DIR}/1_exploracion.png")

    return df


# ---------------------------------------------------------------------------
# 2. Preprocesamiento
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    print("\n" + "=" * 60)
    print("2. PREPROCESAMIENTO")
    print("=" * 60)

    # Codificar variables categoricas con Label Encoding
    df = df.copy()
    le_linea = LabelEncoder()
    df["linea_enc"] = le_linea.fit_transform(df["linea"])
    print(f"\n  Lineas codificadas: {dict(zip(le_linea.classes_, le_linea.transform(le_linea.classes_)))}")

    # Variable objetivo
    le_target = LabelEncoder()
    le_target.fit(CLASS_ORDER)
    y = le_target.transform(df[TARGET_COL])
    print(f"  Target codificado : {dict(zip(CLASS_ORDER, le_target.transform(CLASS_ORDER)))}")

    # Features seleccionadas (no incluimos 'estacion' como string)
    feature_cols = [
        "hora",
        "dia_semana",
        "es_fin_semana",
        "es_hora_pico",
        "linea_enc",
        "posicion_en_linea",
        "es_transferencia",
        "mes",
        "es_festivo",
        "clima",
    ]

    X = df[feature_cols]
    print(f"\n  Features usadas ({len(feature_cols)}): {feature_cols}")

    return X, pd.Series(y, name=TARGET_COL), feature_cols


# ---------------------------------------------------------------------------
# 3. Division entrenamiento / prueba
# ---------------------------------------------------------------------------

def split(X: pd.DataFrame, y: pd.Series):
    print("\n" + "=" * 60)
    print("3. DIVISION ENTRENAMIENTO / PRUEBA")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Entrenamiento: {len(X_train)} muestras ({100*(1-TEST_SIZE):.0f}%)")
    print(f"  Prueba       : {len(X_test)}  muestras ({100*TEST_SIZE:.0f}%)")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. Arbol de Decision con busqueda de hiperparametros
# ---------------------------------------------------------------------------

def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    print("\n" + "=" * 60)
    print("4. ARBOL DE DECISION - BUSQUEDA DE HIPERPARAMETROS")
    print("=" * 60)

    param_grid = {
        "max_depth":        [3, 5, 7, 10, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "criterion":        ["gini", "entropy"],
    }

    base_dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        base_dt, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_dt = grid_search.best_estimator_
    print(f"\n  Mejores hiperparametros : {grid_search.best_params_}")
    print(f"  Accuracy en CV (train)  : {grid_search.best_score_:.4f}")

    return best_dt


# ---------------------------------------------------------------------------
# 5. Evaluacion del modelo
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, feature_cols: list[str], model_name: str) -> float:
    print("\n" + "=" * 60)
    print(f"5. EVALUACION - {model_name}")
    print("=" * 60)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy en prueba: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n  Reporte de clasificacion:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_ORDER))

    # Matriz de confusion
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=CLASS_ORDER,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Matriz de confusion - {model_name}")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(os.path.join(OUTPUT_DIR, f"confusion_{safe_name}.png"), dpi=150)
    plt.close()
    print(f"  -> Grafico guardado: {OUTPUT_DIR}/confusion_{safe_name}.png")

    return acc


# ---------------------------------------------------------------------------
# 6. Importancia de variables
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_cols: list[str], model_name: str) -> None:
    print("\n" + "=" * 60)
    print(f"6. IMPORTANCIA DE VARIABLES - {model_name}")
    print("=" * 60)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n  Ranking de importancia:")
    for rank, i in enumerate(indices, 1):
        print(f"  {rank:2}. {feature_cols[i]:<22} {importances[i]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        [feature_cols[i] for i in reversed(indices)],
        [importances[i] for i in reversed(indices)],
        color="#1976D2",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title(f"Importancia de variables - {model_name}")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_xlim(0, max(importances) * 1.15)
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(os.path.join(OUTPUT_DIR, f"importancia_{safe_name}.png"), dpi=150)
    plt.close()
    print(f"  -> Grafico guardado: {OUTPUT_DIR}/importancia_{safe_name}.png")


# ---------------------------------------------------------------------------
# 7. Visualizacion del arbol de decision
# ---------------------------------------------------------------------------

def visualize_tree(dt: DecisionTreeClassifier, feature_cols: list[str]) -> None:
    print("\n" + "=" * 60)
    print("7. VISUALIZACION DEL ARBOL DE DECISION")
    print("=" * 60)

    # Texto del arbol
    tree_text = export_text(dt, feature_names=feature_cols, max_depth=4)
    print("\n  Estructura del arbol (primeros 4 niveles):\n")
    print(tree_text)

    # Grafico del arbol (primeros 4 niveles para legibilidad)
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        dt,
        feature_names=feature_cols,
        class_names=CLASS_ORDER,
        filled=True,
        max_depth=4,
        ax=ax,
        fontsize=8,
        impurity=False,
        proportion=False,
    )
    ax.set_title("Arbol de Decision - Congestion Transporte Masivo (4 niveles)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "arbol_decision.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Grafico guardado: {OUTPUT_DIR}/arbol_decision.png")


# ---------------------------------------------------------------------------
# 8. Comparacion de modelos
# ---------------------------------------------------------------------------

def compare_models(dt_acc: float, rf_acc: float) -> None:
    print("\n" + "=" * 60)
    print("8. COMPARACION DE MODELOS")
    print("=" * 60)

    models = ["Arbol de Decision", "Bosque Aleatorio"]
    accs   = [dt_acc, rf_acc]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, accs, color=["#1976D2", "#388E3C"], edgecolor="black", width=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparacion de accuracy en conjunto de prueba")
    ax.axhline(y=0.33, color="red", linestyle="--", linewidth=1, label="baseline (azar)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "comparacion_modelos.png"), dpi=150)
    plt.close()

    print(f"\n  Arbol de Decision : {dt_acc:.4f}")
    print(f"  Bosque Aleatorio  : {rf_acc:.4f}")
    winner = "Bosque Aleatorio" if rf_acc > dt_acc else "Arbol de Decision"
    print(f"\n  Mejor modelo      : {winner}")
    print(f"  -> Grafico guardado: {OUTPUT_DIR}/comparacion_modelos.png")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  MODELO DE APRENDIZAJE SUPERVISADO")
    print("  Congestion en Transporte Masivo - Metro de Medellin")
    print("=" * 60 + "\n")

    # 1. Carga y exploracion
    df = load_and_explore(DATA_FILE)

    # 2. Preprocesamiento
    X, y, feature_cols = preprocess(df)

    # 3. Division
    X_train, X_test, y_train, y_test = split(X, y)

    # 4. Arbol de Decision
    dt = train_decision_tree(X_train, y_train)

    # 5. Evaluacion Arbol de Decision
    dt_acc = evaluate(dt, X_test, y_test, feature_cols, "Arbol de Decision")

    # 6. Importancia de variables - Arbol
    plot_feature_importance(dt, feature_cols, "Arbol de Decision")

    # 7. Visualizacion del arbol
    visualize_tree(dt, feature_cols)

    # --- Bosque Aleatorio como comparacion ---
    print("\n" + "=" * 60)
    print("ENTRENANDO BOSQUE ALEATORIO (comparacion)")
    print("=" * 60)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    rf_acc = evaluate(rf, X_test, y_test, feature_cols, "Bosque Aleatorio")
    plot_feature_importance(rf, feature_cols, "Bosque Aleatorio")

    # 8. Comparacion final
    compare_models(dt_acc, rf_acc)

    print("\n" + "=" * 60)
    print("  PROCESO COMPLETADO")
    print(f"  Resultados guardados en: {OUTPUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
