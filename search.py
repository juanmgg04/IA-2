import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

from knowledge_base import STATIONS
from inference_engine import InferenceEngine


@dataclass
class RouteResult:
    origin: str
    destination: str
    total_time: float
    stations: list[str]
    lines: list[str]
    transfers: int
    num_stations: int


@dataclass(order=True)
class _Node:
    f: float
    g: float = field(compare=False)
    station: str = field(compare=False)
    line: str = field(compare=False)
    path: list = field(compare=False, default_factory=list)
    path_lines: list = field(compare=False, default_factory=list)


def _heuristic(a: str, b: str) -> float:
    ax, ay = STATIONS[a]["x"], STATIONS[a]["y"]
    bx, by = STATIONS[b]["x"], STATIONS[b]["y"]
    return math.hypot(ax - bx, ay - by)


def a_star(engine: InferenceEngine, origin: str, destination: str) -> Optional[RouteResult]:
    if not engine.station_exists(origin):
        raise ValueError(f"La estacion '{origin}' no existe en el sistema.")
    if not engine.station_exists(destination):
        raise ValueError(f"La estacion '{destination}' no existe en el sistema.")
    if origin == destination:
        first_line = STATIONS[origin]["lines"][0]
        return RouteResult(origin, destination, 0, [origin], [first_line], 0, 1)

    open_set: list[_Node] = []
    visited: set[tuple[str, str]] = set()

    for line in STATIONS[origin]["lines"]:
        h = _heuristic(origin, destination)
        node = _Node(
            f=h, g=0.0,
            station=origin, line=line,
            path=[origin], path_lines=[line],
        )
        heapq.heappush(open_set, node)

    while open_set:
        current = heapq.heappop(open_set)

        if current.station == destination:
            transfers = sum(
                1 for i in range(1, len(current.path_lines))
                if current.path_lines[i] != current.path_lines[i - 1]
            )
            return RouteResult(
                origin=origin,
                destination=destination,
                total_time=current.g,
                stations=current.path,
                lines=current.path_lines,
                transfers=transfers,
                num_stations=len(current.path),
            )

        state = (current.station, current.line)
        if state in visited:
            continue
        visited.add(state)

        for neighbor, cost, edge_line in engine.neighbors(current.station):
            new_state = (neighbor, edge_line)
            if new_state in visited:
                continue

            penalty = engine.transfer_cost if edge_line != current.line else 0
            new_g = current.g + cost + penalty
            new_h = _heuristic(neighbor, destination)

            node = _Node(
                f=new_g + new_h,
                g=new_g,
                station=neighbor,
                line=edge_line,
                path=current.path + [neighbor],
                path_lines=current.path_lines + [edge_line],
            )
            heapq.heappush(open_set, node)

    return None
