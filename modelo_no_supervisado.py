"""
modelo_no_supervisado.py
========================
Modelos de aprendizaje NO supervisado para descubrir patrones de viaje
en el Metro de Medellin.

Algoritmos aplicados:
  1. K-Means          - agrupamiento por centroides (algoritmo principal)
  2. Jerarquico       - dendrograma para visualizar similitudes
  3. DBSCAN           - agrupamiento por densidad (comparacion)

Flujo:
  1. Carga y exploracion
  2. Preprocesamiento y escalado (StandardScaler)
  3. Reduccion de dimensionalidad (PCA 2D) para visualizacion
  4. Metodo del codo + Silhouette para elegir k optimo
  5. K-Means con k optimo
  6. Perfil de cada cluster (que representa cada grupo)
  7. Clustering jerarquico (dendrograma)
  8. DBSCAN

Uso:
  python modelo_no_supervisado.py

Dependencias:
  pip install pandas scikit-learn matplotlib seaborn scipy
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

DATA_FILE  = "viajes.csv"
OUTPUT_DIR = "resultados_ns"
RANDOM_STATE = 42
ZONE_NAMES = ["Norte", "Centro", "Sur", "Oriente", "Cable"]
CLUSTER_COLORS = ["#4f8ef7", "#f7834f", "#4fcf70", "#f7d24f", "#cf4ff7"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Carga y exploracion
# ---------------------------------------------------------------------------

def load_and_explore(path: str) -> pd.DataFrame:
    print("=" * 60)
    print("1. CARGA Y EXPLORACION DEL DATASET")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"\n  Registros  : {len(df)}")
    print(f"  Variables  : {list(df.columns)}")
    print(f"  Nulos      : {df.isnull().sum().sum()}")
    print(f"\n{df.describe().round(2)}")

    # Distribucion de variables clave
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Distribucion de variables del dataset de viajes", fontsize=13)

    df["hora"].plot.hist(ax=axes[0, 0], bins=18, color="#4f8ef7", edgecolor="black", linewidth=0.4)
    axes[0, 0].set_title("Hora del viaje")

    df["duracion_min"].plot.hist(ax=axes[0, 1], bins=30, color="#f7834f", edgecolor="black", linewidth=0.4)
    axes[0, 1].set_title("Duracion (min)")

    df["ocupacion_tren"].plot.hist(ax=axes[0, 2], bins=25, color="#4fcf70", edgecolor="black", linewidth=0.4)
    axes[0, 2].set_title("Ocupacion del tren (%)")

    df["num_estaciones"].plot.hist(ax=axes[1, 0], bins=20, color="#f7d24f", edgecolor="black", linewidth=0.4)
    axes[1, 0].set_title("Estaciones recorridas")

    zona_counts = df["zona_origen"].map(lambda z: ZONE_NAMES[z]).value_counts()
    zona_counts.plot.bar(ax=axes[1, 1], color="#4f8ef7", edgecolor="black", linewidth=0.4)
    axes[1, 1].set_title("Zona de origen")
    axes[1, 1].tick_params(axis="x", rotation=30)

    df["num_transbordos"].value_counts().sort_index().plot.bar(
        ax=axes[1, 2], color="#cf4ff7", edgecolor="black", linewidth=0.4
    )
    axes[1, 2].set_title("Transbordos")
    axes[1, 2].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "1_exploracion.png"), dpi=150)
    plt.close()
    print(f"\n  -> Grafico guardado: {OUTPUT_DIR}/1_exploracion.png")

    return df


# ---------------------------------------------------------------------------
# 2. Preprocesamiento
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "hora",
    "es_fin_semana",
    "es_hora_pico",
    "zona_origen",
    "zona_destino",
    "num_estaciones",
    "num_transbordos",
    "duracion_min",
    "ocupacion_tren",
    "clima",
]


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    print("\n" + "=" * 60)
    print("2. PREPROCESAMIENTO Y ESCALADO")
    print("=" * 60)

    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n  Features utilizadas ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"  Media post-escala  : {X_scaled.mean(axis=0).round(3)}")
    print(f"  Std  post-escala   : {X_scaled.std(axis=0).round(3)}")

    return X_scaled, scaler


# ---------------------------------------------------------------------------
# 3. PCA 2D
# ---------------------------------------------------------------------------

def apply_pca(X_scaled: np.ndarray) -> np.ndarray:
    print("\n" + "=" * 60)
    print("3. REDUCCION DE DIMENSIONALIDAD (PCA)")
    print("=" * 60)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    print(f"\n  Varianza explicada PC1 : {var[0]:.3f} ({var[0]*100:.1f}%)")
    print(f"  Varianza explicada PC2 : {var[1]:.3f} ({var[1]*100:.1f}%)")
    print(f"  Varianza acumulada     : {sum(var):.3f} ({sum(var)*100:.1f}%)")

    return X_pca


# ---------------------------------------------------------------------------
# 4. Metodo del codo + Silhouette
# ---------------------------------------------------------------------------

def find_optimal_k(X_scaled: np.ndarray) -> int:
    print("\n" + "=" * 60)
    print("4. METODO DEL CODO + COEFICIENTE DE SILHOUETTE")
    print("=" * 60)

    k_range = range(2, 10)
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Grafico
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(k_range), inertias, "o-", color="#4f8ef7", linewidth=2, markersize=7)
    axes[0].set_title("Metodo del codo (Inertia)")
    axes[0].set_xlabel("Numero de clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, "s-", color="#f7834f", linewidth=2, markersize=7)
    axes[1].set_title("Coeficiente de Silhouette")
    axes[1].set_xlabel("Numero de clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, alpha=0.3)

    # Marcar el k optimo
    best_k = list(k_range)[silhouettes.index(max(silhouettes))]
    axes[1].axvline(best_k, color="red", linestyle="--", linewidth=1.5, label=f"k={best_k} (optimo)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "2_codo_silhouette.png"), dpi=150)
    plt.close()

    print(f"\n  k optimo segun Silhouette: k = {best_k}")
    for k, s, i in zip(k_range, silhouettes, inertias):
        marker = " <-- optimo" if k == best_k else ""
        print(f"  k={k}  silhouette={s:.4f}  inertia={i:.0f}{marker}")

    print(f"\n  -> Grafico guardado: {OUTPUT_DIR}/2_codo_silhouette.png")
    return best_k


# ---------------------------------------------------------------------------
# 5. K-Means
# ---------------------------------------------------------------------------

def run_kmeans(X_scaled: np.ndarray, X_pca: np.ndarray, k: int) -> np.ndarray:
    print("\n" + "=" * 60)
    print(f"5. K-MEANS  (k = {k})")
    print("=" * 60)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=15)
    labels = km.fit_predict(X_scaled)
    score  = silhouette_score(X_scaled, labels)

    print(f"\n  Silhouette Score : {score:.4f}")
    print(f"  Inertia          : {km.inertia_:.2f}")

    unique, counts = np.unique(labels, return_counts=True)
    print("\n  Distribucion de clusters:")
    for c, n in zip(unique, counts):
        print(f"    Cluster {c}: {n} viajes ({n/len(labels)*100:.1f}%)")

    # Scatter PCA coloreado por cluster
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(k):
        mask = labels == c
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            s=18, alpha=0.55, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
            label=f"Cluster {c}",
        )
    ax.set_title(f"K-Means (k={k}) — proyeccion PCA 2D")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "3_kmeans_pca.png"), dpi=150)
    plt.close()
    print(f"  -> Grafico guardado: {OUTPUT_DIR}/3_kmeans_pca.png")

    return labels


# ---------------------------------------------------------------------------
# 6. Perfil de clusters
# ---------------------------------------------------------------------------

def profile_clusters(df: pd.DataFrame, labels: np.ndarray, k: int) -> None:
    print("\n" + "=" * 60)
    print("6. PERFIL DE CLUSTERS")
    print("=" * 60)

    df = df.copy()
    df["cluster"] = labels
    df["zona_origen_nombre"]  = df["zona_origen"].map(lambda z: ZONE_NAMES[z])
    df["zona_destino_nombre"] = df["zona_destino"].map(lambda z: ZONE_NAMES[z])

    CLUSTER_LABELS = {
        0: "Viajero hora pico",
        1: "Recreativo fin de semana",
        2: "Estudiante / Academia",
        3: "Turista del Cable",
    }

    profile_cols = [
        "hora", "es_fin_semana", "es_hora_pico",
        "num_estaciones", "num_transbordos", "duracion_min", "ocupacion_tren",
    ]

    means = df.groupby("cluster")[profile_cols].mean().round(2)
    print(f"\n  Medias por cluster:\n{means.to_string()}")

    # Heatmap de perfiles
    fig, ax = plt.subplots(figsize=(11, 4))
    norm_means = (means - means.min()) / (means.max() - means.min() + 1e-9)
    sns.heatmap(
        norm_means,
        annot=means.values,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Valor normalizado"},
    )
    ax.set_title("Perfil de clusters (valores medios)")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Cluster")
    yticks = [f"C{c}" for c in range(k)]
    ax.set_yticklabels(yticks, rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "4_perfil_clusters.png"), dpi=150)
    plt.close()
    print(f"  -> Grafico guardado: {OUTPUT_DIR}/4_perfil_clusters.png")

    # Radar chart
    _radar_chart(means, profile_cols, k)

    # Zona de origen mas frecuente por cluster
    print("\n  Zona de origen predominante por cluster:")
    for c in range(k):
        top_zona = df[df["cluster"] == c]["zona_origen_nombre"].mode()[0]
        top_dest = df[df["cluster"] == c]["zona_destino_nombre"].mode()[0]
        label = CLUSTER_LABELS.get(c, f"Cluster {c}")
        print(f"    C{c} ({label}): {top_zona} -> {top_dest}")


def _radar_chart(means: pd.DataFrame, cols: list, k: int) -> None:
    """Grafico tipo radar para comparar perfiles de clusters."""
    norm = (means - means.min()) / (means.max() - means.min() + 1e-9)
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for c in range(k):
        values = norm.iloc[c].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], label=f"Cluster {c}")
        ax.fill(angles, values, alpha=0.12,
                color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)])

    ax.set_xticks(angles[:-1])
    short_labels = ["hora", "fin_sem", "h_pico", "estac", "transbordos", "duracion", "ocupacion"]
    ax.set_xticklabels(short_labels, size=9)
    ax.set_title("Radar de perfiles por cluster", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "5_radar_clusters.png"), dpi=150)
    plt.close()
    print(f"  -> Grafico guardado: {OUTPUT_DIR}/5_radar_clusters.png")


# ---------------------------------------------------------------------------
# 7. Clustering jerarquico
# ---------------------------------------------------------------------------

def hierarchical_clustering(X_scaled: np.ndarray) -> None:
    print("\n" + "=" * 60)
    print("7. CLUSTERING JERARQUICO (Dendrograma)")
    print("=" * 60)

    sample_idx = np.random.choice(len(X_scaled), size=300, replace=False)
    X_sample = X_scaled[sample_idx]

    Z = linkage(X_sample, method="ward")

    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(
        Z, ax=ax,
        truncate_mode="lastp", p=30,
        leaf_rotation=90, leaf_font_size=8,
        color_threshold=0.7 * max(Z[:, 2]),
        above_threshold_color="gray",
    )
    ax.set_title("Dendrograma — Clustering Jerarquico (Ward, muestra 300)")
    ax.set_xlabel("Indice de muestra")
    ax.set_ylabel("Distancia (Ward)")
    ax.axhline(y=0.7 * max(Z[:, 2]), color="red", linestyle="--",
               linewidth=1.2, label="Umbral de corte")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "6_dendrograma.png"), dpi=150)
    plt.close()
    print(f"  -> Grafico guardado: {OUTPUT_DIR}/6_dendrograma.png")
    print("  El dendrograma muestra la similitud entre grupos de viajes.")
    print("  Las ramas que se unen a baja altura son mas similares entre si.")


# ---------------------------------------------------------------------------
# 8. DBSCAN
# ---------------------------------------------------------------------------

def run_dbscan(X_scaled: np.ndarray, X_pca: np.ndarray) -> None:
    print("\n" + "=" * 60)
    print("8. DBSCAN (agrupamiento por densidad)")
    print("=" * 60)

    db = DBSCAN(eps=0.8, min_samples=15)
    labels_db = db.fit_predict(X_scaled)

    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise    = list(labels_db).count(-1)

    print(f"\n  Clusters encontrados : {n_clusters}")
    print(f"  Puntos ruido (-1)    : {n_noise} ({n_noise/len(labels_db)*100:.1f}%)")

    if n_clusters > 1:
        mask_valid = labels_db != -1
        score = silhouette_score(X_scaled[mask_valid], labels_db[mask_valid])
        print(f"  Silhouette Score     : {score:.4f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    unique_labels = sorted(set(labels_db))
    palette = ["#aaaaaa"] + CLUSTER_COLORS  # gris para ruido

    for i, lbl in enumerate(unique_labels):
        mask = labels_db == lbl
        color = "#aaaaaa" if lbl == -1 else CLUSTER_COLORS[(lbl) % len(CLUSTER_COLORS)]
        label = "Ruido" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=15, alpha=0.5, color=color, label=label)

    ax.set_title("DBSCAN — proyeccion PCA 2D")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "7_dbscan.png"), dpi=150)
    plt.close()
    print(f"  -> Grafico guardado: {OUTPUT_DIR}/7_dbscan.png")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  MODELO DE APRENDIZAJE NO SUPERVISADO")
    print("  Patrones de viaje - Metro de Medellin")
    print("=" * 60 + "\n")

    df = load_and_explore(DATA_FILE)

    X_scaled, scaler = preprocess(df)
    X_pca = apply_pca(X_scaled)

    best_k = find_optimal_k(X_scaled)
    labels = run_kmeans(X_scaled, X_pca, best_k)

    profile_clusters(df, labels, best_k)
    hierarchical_clustering(X_scaled)
    run_dbscan(X_scaled, X_pca)

    print("\n" + "=" * 60)
    print("  PROCESO COMPLETADO")
    print(f"  Graficos guardados en: {OUTPUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
