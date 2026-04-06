from knowledge_base import RULES, RAW_CONNECTIONS, STATIONS


class InferenceEngine:
    def __init__(self, verbose: bool = True):
        self.graph: dict[str, list[tuple[str, int, str]]] = {}
        self.transfer_stations: set[str] = set()
        self.transfer_cost: int = 0
        self.verbose = verbose
        self._run()

    def _run(self) -> None:
        if self.verbose:
            print("Motor de inferencia - encadenamiento hacia adelante")
            print("-" * 50)

        for rule in RULES:
            self._apply_rule(rule)

        if self.verbose:
            print("-" * 50)
            print(f"Grafo construido: {len(self.graph)} nodos, "
                  f"{sum(len(v) for v in self.graph.values())} aristas")
            print(f"Estaciones de transferencia: {sorted(self.transfer_stations)}")
            print(f"Costo de transbordo        : {self.transfer_cost} minutos\n")

    def _apply_rule(self, rule) -> None:
        if self.verbose:
            print(f"  [{rule.id}] {rule.name}")

        result = rule.apply(self.graph, RAW_CONNECTIONS)

        if rule.id == "R2" and isinstance(result, set):
            self.transfer_stations = result
        elif rule.id == "R3" and isinstance(result, int):
            self.transfer_cost = result

    def neighbors(self, station: str) -> list[tuple[str, int, str]]:
        return self.graph.get(station, [])

    def is_transfer_station(self, station: str) -> bool:
        return station in self.transfer_stations

    def station_exists(self, name: str) -> bool:
        return name in STATIONS

    def all_stations(self) -> list[str]:
        return sorted(STATIONS.keys())

    def stations_by_line(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for name, data in STATIONS.items():
            for line in data["lines"]:
                result.setdefault(line, []).append(name)
        return result
