import sys

from inference_engine import InferenceEngine
from search import a_star, RouteResult


def _separator(char: str = "-", width: int = 55) -> None:
    print(char * width)


def print_header() -> None:
    _separator("=")
    print("  SISTEMA INTELIGENTE DE RUTAS")
    print("  Transporte Masivo - Metro de Medellin")
    print("  Reglas Logicas + Busqueda Heuristica A*")
    _separator("=")
    print()


def print_stations(engine: InferenceEngine) -> None:
    print("ESTACIONES POR LINEA:")
    for line, stations in sorted(engine.stations_by_line().items()):
        label = f"  Linea {line}"
        print(f"{label}: {', '.join(stations)}")
    print()


def print_rules() -> None:
    from knowledge_base import RULES
    print("BASE DE CONOCIMIENTO - REGLAS LOGICAS:")
    for rule in RULES:
        print(f"  [{rule.id}] {rule.name}")
        print(f"       {rule.description}")
    print()


def _format_route(result: RouteResult) -> str:
    lines_out = []
    for i, station in enumerate(result.stations):
        line = result.lines[i]
        if i > 0 and result.lines[i] != result.lines[i - 1]:
            lines_out.append(f"       -- TRANSBORDO a Linea {line} --")
        prefix = "  [INICIO]" if i == 0 else ("  [FINAL] " if i == len(result.stations) - 1 else "         ")
        lines_out.append(f"{prefix} [{line}] {station}")
    return "\n".join(lines_out)


def print_result(result: RouteResult) -> None:
    print(f"Ruta: {result.origin}  ->  {result.destination}")
    _separator()
    print(_format_route(result))
    _separator()
    print(f"  Tiempo estimado : {result.total_time:.0f} minutos")
    print(f"  Estaciones      : {result.num_stations}")
    print(f"  Transbordos     : {result.transfers}")
    print()


TEST_CASES: list[tuple[str, str]] = [
    ("Niquia",      "La Estrella"),
    ("Niquia",      "Trinidad"),
    ("Poblado",     "Santo Domingo"),
    ("La Estrella", "Arvi"),
    ("Trinidad",    "Arvi"),
]


def run_demo(engine: InferenceEngine) -> None:
    print("CASOS DE PRUEBA")
    _separator("=")
    print()

    for origin, destination in TEST_CASES:
        result = a_star(engine, origin, destination)
        if result:
            print_result(result)
        else:
            print(f"  Sin ruta: {origin} -> {destination}\n")


def run_interactive(engine: InferenceEngine) -> None:
    _separator("=")
    print("  MODO INTERACTIVO")
    _separator("=")
    print("  Escriba 'lista' para ver las estaciones disponibles.")
    print("  Escriba 'salir' para terminar.")
    print()

    while True:
        origin = input("Estacion de origen  : ").strip()
        if origin.lower() == "salir":
            break
        if origin.lower() == "lista":
            print_stations(engine)
            continue

        destination = input("Estacion de destino : ").strip()
        if destination.lower() == "salir":
            break

        print()
        try:
            result = a_star(engine, origin, destination)
            if result:
                print_result(result)
            else:
                print(f"  No se encontro ruta entre '{origin}' y '{destination}'.\n")
        except ValueError as exc:
            print(f"  Error: {exc}\n")


def main() -> None:
    args = sys.argv[1:]
    demo_only        = "--demo"        in args
    interactive_only = "--interactive" in args

    print_header()

    engine = InferenceEngine(verbose=True)

    print_rules()
    print_stations(engine)

    if not interactive_only:
        run_demo(engine)

    if not demo_only:
        run_interactive(engine)


if __name__ == "__main__":
    main()
