from dataclasses import dataclass
from typing import Callable


STATIONS: dict[str, dict] = {
    "Niquia":        {"lines": ["A"],      "x": 0.0,  "y": 10.0},
    "Bello":         {"lines": ["A"],      "x": 0.0,  "y":  9.0},
    "Madera":        {"lines": ["A"],      "x": 0.0,  "y":  8.0},
    "Acevedo":       {"lines": ["A", "K"], "x": 0.0,  "y":  7.0},
    "Tricentenario": {"lines": ["A"],      "x": 0.0,  "y":  6.0},
    "Caribe":        {"lines": ["A"],      "x": 0.0,  "y":  5.0},
    "Universidad":   {"lines": ["A"],      "x": 0.0,  "y":  4.0},
    "Hospital":      {"lines": ["A"],      "x": 0.0,  "y":  3.0},
    "Prado":         {"lines": ["A"],      "x": 0.0,  "y":  2.0},
    "Parque Berrio": {"lines": ["A"],      "x": 0.0,  "y":  1.0},
    "San Antonio":   {"lines": ["A", "B"], "x": 0.0,  "y":  0.0},
    "Alpujarra":     {"lines": ["A"],      "x": 0.0,  "y": -1.0},
    "Exposiciones":  {"lines": ["A"],      "x": 0.0,  "y": -2.0},
    "Industriales":  {"lines": ["A"],      "x": 0.0,  "y": -3.0},
    "Poblado":       {"lines": ["A"],      "x": 0.0,  "y": -4.0},
    "Aguacatala":    {"lines": ["A"],      "x": 0.0,  "y": -5.0},
    "Ayura":         {"lines": ["A"],      "x": 0.0,  "y": -6.0},
    "Envigado":      {"lines": ["A"],      "x": 0.0,  "y": -7.0},
    "Itagui":        {"lines": ["A"],      "x": 0.0,  "y": -8.0},
    "La Estrella":   {"lines": ["A"],      "x": 0.0,  "y": -9.0},

    "Cisneros":      {"lines": ["B"],      "x": 1.0,  "y":  0.0},
    "Suramericana":  {"lines": ["B"],      "x": 2.0,  "y":  0.0},
    "Estadio":       {"lines": ["B"],      "x": 3.0,  "y":  0.0},
    "Floresta":      {"lines": ["B"],      "x": 4.0,  "y":  0.0},
    "Santa Lucia":   {"lines": ["B"],      "x": 5.0,  "y":  0.0},
    "Trinidad":      {"lines": ["B"],      "x": 6.0,  "y":  0.0},

    "Andalucia":     {"lines": ["K"],      "x": 0.5,  "y":  7.5},
    "Popular":       {"lines": ["K"],      "x": 1.0,  "y":  8.0},
    "Santo Domingo": {"lines": ["K"],      "x": 1.5,  "y":  8.5},
    "Arvi":          {"lines": ["K"],      "x": 2.0,  "y":  9.5},
}

RAW_CONNECTIONS: list[tuple[str, str, int, str]] = [
    ("Niquia",        "Bello",          3, "A"),
    ("Bello",         "Madera",         2, "A"),
    ("Madera",        "Acevedo",        2, "A"),
    ("Acevedo",       "Tricentenario",  2, "A"),
    ("Tricentenario", "Caribe",         2, "A"),
    ("Caribe",        "Universidad",    2, "A"),
    ("Universidad",   "Hospital",       1, "A"),
    ("Hospital",      "Prado",          1, "A"),
    ("Prado",         "Parque Berrio",  1, "A"),
    ("Parque Berrio", "San Antonio",    1, "A"),
    ("San Antonio",   "Alpujarra",      2, "A"),
    ("Alpujarra",     "Exposiciones",   2, "A"),
    ("Exposiciones",  "Industriales",   2, "A"),
    ("Industriales",  "Poblado",        3, "A"),
    ("Poblado",       "Aguacatala",     2, "A"),
    ("Aguacatala",    "Ayura",          2, "A"),
    ("Ayura",         "Envigado",       2, "A"),
    ("Envigado",      "Itagui",         3, "A"),
    ("Itagui",        "La Estrella",    3, "A"),

    ("San Antonio",   "Cisneros",       2, "B"),
    ("Cisneros",      "Suramericana",   2, "B"),
    ("Suramericana",  "Estadio",        2, "B"),
    ("Estadio",       "Floresta",       2, "B"),
    ("Floresta",      "Santa Lucia",    3, "B"),
    ("Santa Lucia",   "Trinidad",       3, "B"),

    ("Acevedo",       "Andalucia",      4, "K"),
    ("Andalucia",     "Popular",        4, "K"),
    ("Popular",       "Santo Domingo",  4, "K"),
    ("Santo Domingo", "Arvi",           8, "K"),
]

TRANSFER_COST: int = 3


@dataclass
class Rule:
    id: str
    name: str
    description: str
    apply: Callable


def _r1_bidirectional(graph: dict, _raw: list) -> None:
    for origin, dest, cost, line in RAW_CONNECTIONS:
        graph.setdefault(origin, []).append((dest,   cost, line))
        graph.setdefault(dest,   []).append((origin, cost, line))


def _r2_transfer_stations(_graph: dict, _raw: list) -> set[str]:
    return {name for name, data in STATIONS.items() if len(data["lines"]) > 1}


def _r3_transfer_cost(_graph: dict, _raw: list) -> int:
    return TRANSFER_COST


def _r4_valid_route(path: list[str]) -> bool:
    return all(station in STATIONS for station in path)


RULES: list[Rule] = [
    Rule(
        id="R1",
        name="Conexion bidireccional",
        description=(
            "Si existe conexion(A, B, C, L) entonces existe conexion(B, A, C, L). "
            "El tren circula en ambas direcciones."
        ),
        apply=_r1_bidirectional,
    ),
    Rule(
        id="R2",
        name="Estacion de transferencia",
        description=(
            "Si pertenece(E, L1) AND pertenece(E, L2) AND L1 != L2 "
            "entonces es_transferencia(E)."
        ),
        apply=_r2_transfer_stations,
    ),
    Rule(
        id="R3",
        name="Penalizacion por transbordo",
        description=(
            "Si cambia_linea(pasajero, L1, L2) en estacion de transferencia "
            "entonces costo += TRANSFER_COST."
        ),
        apply=_r3_transfer_cost,
    ),
    Rule(
        id="R4",
        name="Validacion de ruta",
        description=(
            "Una ruta es valida si todas las estaciones existen "
            "y cada par consecutivo esta conectado."
        ),
        apply=lambda *_: None,
    ),
]
