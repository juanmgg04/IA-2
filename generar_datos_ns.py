"""
generar_datos_ns.py
===================
Genera viajes.csv: 2 500 registros de viajes realizados en el Metro
de Medellin. No incluye variable objetivo (aprendizaje NO supervisado).

Cada fila representa un viaje individual con sus caracteristicas
observables. Los patrones naturales que deben emerger del clustering son:

  - Viajeros de hora pico  : dias laborales, horas punta, recorridos largos
  - Recreativos            : fines de semana, recorridos medios, baja ocupacion
  - Estudiantes / academia : mananas laborales, zona centro
  - Turistas del cable     : zona K (Metrocable), recorridos cortos

Variables generadas:
  hora              int  (5-22)
  dia_semana        int  (0=lun ... 6=dom)
  es_fin_semana     int  (0/1)
  es_hora_pico      int  (0/1)  horas 7-9 y 17-19
  zona_origen       int  (0=Norte 1=Centro 2=Sur 3=Oriente 4=Cable)
  zona_destino      int  (0-4)
  num_estaciones    int  estaciones recorridas
  num_transbordos   int  (0-2)
  duracion_min      float minutos totales del viaje
  ocupacion_tren    float porcentaje de ocupacion (0-100)
  clima             int  (0=soleado 1=nublado 2=lluvia)

Uso:
  python generar_datos_ns.py   ->  genera viajes.csv
"""

import csv
import random
import math

SEED = 7
random.seed(SEED)

OUTPUT_FILE = "viajes.csv"
NUM_SAMPLES = 2_500

# Zonas del sistema
# 0-Norte  1-Centro  2-Sur  3-Oriente  4-Cable
ZONE_NAMES = ["Norte", "Centro", "Sur", "Oriente", "Cable"]

# Tiempo aproximado entre zonas (minutos, triangular superior)
# Distancias simétricas
ZONE_TRAVEL = [
    [0,  8, 20, 12, 18],   # Norte
    [8,  0, 14,  8, 16],   # Centro
    [20, 14,  0, 18, 28],  # Sur
    [12,  8, 18,  0, 22],  # Oriente
    [18, 16, 28, 22,  0],  # Cable
]

# Numero de estaciones aproximado entre zonas
ZONE_STATIONS = [
    [0,  6, 14,  8, 12],
    [6,  0,  8,  5, 10],
    [14, 8,  0, 12, 18],
    [8,  5, 12,  0, 14],
    [12, 10, 18, 14,  0],
]


def _gauss_clamp(mu: float, sigma: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, random.gauss(mu, sigma)))


def _generate_peak_commuter() -> dict:
    """Patron: viajero hora pico, dias laborales, recorrido largo."""
    hora = random.choice([7, 7, 8, 8, 8, 9, 17, 17, 18, 18, 18, 19])
    dia  = random.randint(0, 4)                    # lun-vie
    zona_o = random.choice([0, 2, 3])              # periferia
    zona_d = 1                                     # centro
    if random.random() < 0.4:
        zona_o, zona_d = zona_d, zona_o            # regreso al hogar

    base_dur = ZONE_TRAVEL[zona_o][zona_d]
    base_est = ZONE_STATIONS[zona_o][zona_d]
    transbordos = 1 if (zona_o in (3, 4) or zona_d in (3, 4)) else 0
    duracion = _gauss_clamp(base_dur + transbordos * 3, 4, 5, 65)
    estaciones = max(2, int(_gauss_clamp(base_est, 2, 2, 20)))
    ocupacion = _gauss_clamp(82, 10, 40, 100)

    return dict(
        hora=hora, dia_semana=dia, es_fin_semana=0, es_hora_pico=1,
        zona_origen=zona_o, zona_destino=zona_d,
        num_estaciones=estaciones, num_transbordos=transbordos,
        duracion_min=round(duracion, 1), ocupacion_tren=round(ocupacion, 1),
        clima=random.choices([0, 1, 2], weights=[.55, .30, .15])[0],
    )


def _generate_recreational() -> dict:
    """Patron: viaje recreativo, fin de semana, recorrido variado."""
    hora = random.randint(9, 19)
    dia  = random.choice([5, 6])                   # sab-dom
    zona_o = random.randint(0, 4)
    zona_d = random.randint(0, 4)
    while zona_d == zona_o:
        zona_d = random.randint(0, 4)

    base_dur = ZONE_TRAVEL[zona_o][zona_d]
    base_est = ZONE_STATIONS[zona_o][zona_d]
    transbordos = 1 if (zona_o in (3, 4) or zona_d in (3, 4)) else 0
    duracion = _gauss_clamp(base_dur + transbordos * 3, 5, 3, 60)
    estaciones = max(1, int(_gauss_clamp(base_est, 3, 1, 20)))
    ocupacion = _gauss_clamp(45, 15, 10, 85)

    return dict(
        hora=hora, dia_semana=dia, es_fin_semana=1, es_hora_pico=0,
        zona_origen=zona_o, zona_destino=zona_d,
        num_estaciones=estaciones, num_transbordos=transbordos,
        duracion_min=round(duracion, 1), ocupacion_tren=round(ocupacion, 1),
        clima=random.choices([0, 1, 2], weights=[.55, .30, .15])[0],
    )


def _generate_student() -> dict:
    """Patron: estudiante, mananas laborales, zona centro/universidad."""
    hora = random.choice([6, 6, 7, 7, 8, 10, 11, 12])
    dia  = random.randint(0, 4)
    zona_o = random.choice([0, 0, 2, 3])
    zona_d = 1                                     # Centro (Universidad)

    base_dur = ZONE_TRAVEL[zona_o][zona_d]
    base_est = ZONE_STATIONS[zona_o][zona_d]
    transbordos = 0
    duracion = _gauss_clamp(base_dur, 3, 3, 40)
    estaciones = max(1, int(_gauss_clamp(base_est, 2, 1, 14)))
    ocupacion = _gauss_clamp(60, 12, 20, 90)

    return dict(
        hora=hora, dia_semana=dia, es_fin_semana=0,
        es_hora_pico=1 if hora in (7, 8) else 0,
        zona_origen=zona_o, zona_destino=zona_d,
        num_estaciones=estaciones, num_transbordos=transbordos,
        duracion_min=round(duracion, 1), ocupacion_tren=round(ocupacion, 1),
        clima=random.choices([0, 1, 2], weights=[.55, .30, .15])[0],
    )


def _generate_cable_tourist() -> dict:
    """Patron: turista del Metrocable, recorrido corto, baja ocupacion."""
    hora = random.randint(8, 16)
    dia  = random.randint(0, 6)
    zona_o = random.choice([1, 0, 4])
    zona_d = 4                                     # Cable
    if random.random() < 0.5:
        zona_o, zona_d = zona_d, zona_o

    base_dur = ZONE_TRAVEL[zona_o][zona_d]
    base_est = ZONE_STATIONS[zona_o][zona_d]
    transbordos = 1 if zona_o != 4 and zona_d != 4 else 0
    duracion = _gauss_clamp(base_dur + transbordos * 3, 5, 5, 50)
    estaciones = max(1, int(_gauss_clamp(base_est, 2, 1, 16)))
    ocupacion = _gauss_clamp(35, 12, 5, 70)

    return dict(
        hora=hora, dia_semana=dia,
        es_fin_semana=1 if dia >= 5 else 0,
        es_hora_pico=0,
        zona_origen=zona_o, zona_destino=zona_d,
        num_estaciones=estaciones, num_transbordos=transbordos,
        duracion_min=round(duracion, 1), ocupacion_tren=round(ocupacion, 1),
        clima=random.choices([0, 1, 2], weights=[.60, .28, .12])[0],
    )


GENERATORS = [
    (_generate_peak_commuter, 0.38),
    (_generate_recreational,  0.28),
    (_generate_student,       0.22),
    (_generate_cable_tourist, 0.12),
]


def generate(n: int) -> list[dict]:
    records = []
    funcs, weights = zip(*GENERATORS)
    for _ in range(n):
        gen = random.choices(funcs, weights=weights)[0]
        records.append(gen())
    return records


def save_csv(records: list[dict], path: str) -> None:
    fields = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def print_summary(records: list[dict]) -> None:
    print(f"Dataset generado : {len(records)} registros  ->  {OUTPUT_FILE}")
    horas = [r["hora"] for r in records]
    print(f"Hora min/max     : {min(horas)} / {max(horas)}")
    pico = sum(r["es_hora_pico"] for r in records)
    fds  = sum(r["es_fin_semana"] for r in records)
    print(f"Hora pico        : {pico} ({pico/len(records)*100:.1f}%)")
    print(f"Fin de semana    : {fds}  ({fds/len(records)*100:.1f}%)")
    cable = sum(1 for r in records if r["zona_origen"] == 4 or r["zona_destino"] == 4)
    print(f"Viajes con Cable : {cable} ({cable/len(records)*100:.1f}%)")


if __name__ == "__main__":
    data = generate(NUM_SAMPLES)
    save_csv(data, OUTPUT_FILE)
    print_summary(data)
