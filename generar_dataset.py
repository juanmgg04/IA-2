"""
generar_dataset.py
==================
Genera un dataset sintetico pero realista sobre el nivel de congestion
en las estaciones del Metro de Medellin.

Cada fila representa una observacion de una estacion en un momento dado.

Variables de entrada (features):
  hora               : hora del dia (0-23)
  dia_semana         : 0=lunes ... 6=domingo
  es_fin_semana      : 1 si sabado o domingo
  es_hora_pico       : 1 si hora pico (7-9h o 17-19h)
  linea              : linea del sistema (A, B, K)
  posicion_en_linea  : posicion relativa de la estacion en su linea (0.0-1.0)
  es_transferencia   : 1 si es estacion de transbordo
  mes                : mes del ano (1-12)
  es_festivo         : 1 si es dia festivo en Colombia
  clima              : 0=soleado, 1=nublado, 2=lluvia

Variable objetivo (target):
  nivel_congestion   : bajo / medio / alto

Uso:
  python generar_dataset.py
  -> genera datos_transporte.csv con 3 000 registros
"""

import csv
import random

SEED = 42
random.seed(SEED)

OUTPUT_FILE = "datos_transporte.csv"
NUM_SAMPLES = 3_000

# ---------------------------------------------------------------------------
# Definicion de estaciones por linea
# ---------------------------------------------------------------------------

STATIONS_BY_LINE: dict[str, list[str]] = {
    "A": [
        "Niquia", "Bello", "Madera", "Acevedo", "Tricentenario",
        "Caribe", "Universidad", "Hospital", "Prado", "Parque Berrio",
        "San Antonio", "Alpujarra", "Exposiciones", "Industriales",
        "Poblado", "Aguacatala", "Ayura", "Envigado", "Itagui", "La Estrella",
    ],
    "B": [
        "San Antonio", "Cisneros", "Suramericana", "Estadio",
        "Floresta", "Santa Lucia", "Trinidad",
    ],
    "K": [
        "Acevedo", "Andalucia", "Popular", "Santo Domingo", "Arvi",
    ],
}

TRANSFER_STATIONS = {"San Antonio", "Acevedo"}

# Festivos aproximados en Colombia (mes, dia)
FESTIVOS = {
    (1, 1), (1, 6), (3, 24), (4, 1), (4, 17), (4, 18),
    (5, 1), (6, 2), (6, 23), (6, 30), (7, 20), (8, 7),
    (8, 18), (10, 13), (11, 3), (11, 17), (12, 8), (12, 25),
}

# ---------------------------------------------------------------------------
# Logica de generacion del nivel de congestion
# ---------------------------------------------------------------------------

def _score_congestion(
    hora: int,
    dia_semana: int,
    es_festivo: int,
    linea: str,
    pos_relativa: float,
    es_transferencia: int,
    mes: int,
    clima: int,
) -> float:
    """
    Calcula una puntuacion de congestion continua basada en reglas
    derivadas del dominio de transporte masivo.
    """
    score = 0.0

    # --- Efecto horario ---
    if hora in (7, 8, 9, 17, 18, 19):     # hora pico
        score += 3.5
    elif hora in (6, 10, 16, 20):          # hora semi-pico
        score += 2.0
    elif 11 <= hora <= 15:                  # valle del medio dia
        score += 1.0
    elif hora in (0, 1, 2, 3, 4, 5):       # madrugada
        score -= 2.0

    # --- Efecto dia de la semana ---
    if dia_semana < 5 and not es_festivo:   # dia laboral
        score += 1.5
    elif dia_semana == 5:                   # sabado
        score -= 0.5
    else:                                   # domingo o festivo
        score -= 1.5

    # --- Efecto linea ---
    if linea == "A":                        # linea principal, mayor afluencia
        score += 1.0
    elif linea == "K":                      # cable, menor capacidad y demanda
        score -= 0.5

    # --- Efecto posicion en la linea ---
    # Las estaciones centrales (posicion ~0.5) tienen mas afluencia
    centrality = 1.0 - abs(pos_relativa - 0.5) * 2
    score += centrality * 1.2

    # --- Estacion de transferencia ---
    if es_transferencia:
        score += 2.0

    # --- Efecto mensual (vacaciones escolares: jun, jul, dic) ---
    if mes in (6, 7, 12):
        score -= 0.8

    # --- Efecto clima ---
    if clima == 2:                          # lluvia: mas gente usa el metro
        score += 0.8
    elif clima == 1:                        # nublado: efecto leve
        score += 0.3

    # Ruido gaussiano para simular variabilidad real
    score += random.gauss(0, 0.9)

    return score


def _label(score: float) -> str:
    if score <= 1.8:
        return "bajo"
    elif score <= 3.8:
        return "medio"
    else:
        return "alto"


# ---------------------------------------------------------------------------
# Generacion del dataset
# ---------------------------------------------------------------------------

def generate_dataset(n: int) -> list[dict]:
    records = []

    for _ in range(n):
        # Seleccion aleatoria de contexto
        linea = random.choice(["A", "A", "A", "B", "B", "K"])  # A mas frecuente
        estaciones = STATIONS_BY_LINE[linea]
        idx = random.randint(0, len(estaciones) - 1)
        estacion = estaciones[idx]
        pos_relativa = idx / max(len(estaciones) - 1, 1)

        hora = random.randint(5, 22)
        dia_semana = random.randint(0, 6)
        mes = random.randint(1, 12)
        dia = random.randint(1, 28)
        es_festivo = 1 if (mes, dia) in FESTIVOS else 0
        clima = random.choices([0, 1, 2], weights=[0.55, 0.30, 0.15])[0]
        es_transferencia = 1 if estacion in TRANSFER_STATIONS else 0
        es_fin_semana = 1 if dia_semana >= 5 else 0
        es_hora_pico = 1 if hora in (7, 8, 9, 17, 18, 19) else 0

        score = _score_congestion(
            hora, dia_semana, es_festivo,
            linea, pos_relativa, es_transferencia,
            mes, clima,
        )

        records.append({
            "hora":               hora,
            "dia_semana":         dia_semana,
            "es_fin_semana":      es_fin_semana,
            "es_hora_pico":       es_hora_pico,
            "linea":              linea,
            "estacion":           estacion,
            "posicion_en_linea":  round(pos_relativa, 3),
            "es_transferencia":   es_transferencia,
            "mes":                mes,
            "es_festivo":         es_festivo,
            "clima":              clima,
            "nivel_congestion":   _label(score),
        })

    return records


def save_csv(records: list[dict], path: str) -> None:
    fields = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def print_summary(records: list[dict]) -> None:
    from collections import Counter
    counts = Counter(r["nivel_congestion"] for r in records)
    total = len(records)
    print(f"Dataset generado: {total} registros  ->  {OUTPUT_FILE}")
    print("Distribucion de clases:")
    for label in ("bajo", "medio", "alto"):
        n = counts[label]
        pct = n / total * 100
        print(f"  {label:<6}: {n:>5} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = generate_dataset(NUM_SAMPLES)
    save_csv(data, OUTPUT_FILE)
    print_summary(data)
