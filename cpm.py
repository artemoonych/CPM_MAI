"""
CPM-анализ проекта цифровизации образовательного учреждения.

Данные проекта вынесены в отдельный JSON-файл activities.json.

Программа выполняет:
1. Загрузку исходных данных из внешнего файла.
2. Формирование сетевой модели проекта.
3. Проверку корректности зависимостей.
4. Расчёт ранних сроков начала и окончания работ.
5. Расчёт поздних сроков начала и окончания работ.
6. Расчёт полного резерва времени.
7. Определение критических работ и критических путей.
8. Построение сетевого графика.
9. Построение диаграммы Ганта. 

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


@dataclass(frozen=True)
class Activity:
    code: str
    name: str
    duration: int
    predecessors: List[str]

def load_activities_from_json(file_path: str | Path) -> Dict[str, Activity]:
    """Загружает список работ проекта из JSON-файла."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Файл с исходными данными не найден: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if "activities" not in raw_data:
        raise ValueError("В JSON-файле должен быть ключ 'activities'.")

    activities: Dict[str, Activity] = {}

    for item in raw_data["activities"]:
        required_fields = {"code", "name", "duration", "predecessors"}
        missing_fields = required_fields - set(item.keys())

        if missing_fields:
            raise ValueError(
                f"В записи работы отсутствуют поля: {', '.join(missing_fields)}"
            )

        code = str(item["code"]).strip()
        name = str(item["name"]).strip()
        duration = int(item["duration"])
        predecessors = list(item["predecessors"])

        if not code:
            raise ValueError("Код работы не может быть пустым.")

        if code in activities:
            raise ValueError(f"Код работы {code} повторяется в файле данных.")

        activities[code] = Activity(
            code=code,
            name=name,
            duration=duration,
            predecessors=predecessors,
        )

    return activities

def build_graph(activities: Dict[str, Activity]) -> nx.DiGraph:
    """Создаёт ориентированный граф зависимостей работ."""

    graph = nx.DiGraph()

    for code, activity in activities.items():
        if activity.duration <= 0:
            raise ValueError(f"Работа {code} имеет некорректную длительность")

        graph.add_node(
            code,
            name=activity.name,
            duration=activity.duration,
        )

    for code, activity in activities.items():
        for predecessor in activity.predecessors:
            if predecessor not in activities:
                raise ValueError(
                    f"Для работы {code} указан несуществующий предшественник {predecessor}"
                )
            graph.add_edge(predecessor, code)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Сетевой график содержит цикл. CPM требует ациклический граф.")

    return graph

def calculate_cpm(
    graph: nx.DiGraph,
    activities: Dict[str, Activity],
) -> Tuple[List[dict], int]:
    """Выполняет прямой и обратный проход CPM."""

    topological_order = list(nx.topological_sort(graph))

    early_start: Dict[str, int] = {}
    early_finish: Dict[str, int] = {}

    # Прямой проход: расчёт ранних сроков
    for code in topological_order:
        predecessors = list(graph.predecessors(code))

        if predecessors:
            early_start[code] = max(early_finish[pred] for pred in predecessors)
        else:
            early_start[code] = 0

        early_finish[code] = early_start[code] + activities[code].duration

    project_duration = max(early_finish.values())

    late_start: Dict[str, int] = {}
    late_finish: Dict[str, int] = {}

    for code in reversed(topological_order):
        successors = list(graph.successors(code))

        if successors:
            late_finish[code] = min(late_start[succ] for succ in successors)
        else:
            late_finish[code] = project_duration

        late_start[code] = late_finish[code] - activities[code].duration

    result: List[dict] = []

    for code in topological_order:
        total_float = late_start[code] - early_start[code]
        is_critical = total_float == 0

        result.append(
            {
                "code": code,
                "name": activities[code].name,
                "duration": activities[code].duration,
                "predecessors": activities[code].predecessors,
                "ES": early_start[code],
                "EF": early_finish[code],
                "LS": late_start[code],
                "LF": late_finish[code],
                "float": total_float,
                "is_critical": is_critical,
            }
        )

    return result, project_duration

def find_critical_paths(
    graph: nx.DiGraph,
    cpm_result: List[dict],
) -> List[List[str]]:
    """Находит все критические пути в сетевом графике."""

    critical_nodes = {
        row["code"]
        for row in cpm_result
        if row["is_critical"]
    }

    critical_graph = graph.subgraph(critical_nodes).copy()

    sources = [node for node in critical_graph.nodes if critical_graph.in_degree(node) == 0]
    sinks = [node for node in critical_graph.nodes if critical_graph.out_degree(node) == 0]

    critical_paths: List[List[str]] = []

    for source in sources:
        for sink in sinks:
            for path in nx.all_simple_paths(critical_graph, source=source, target=sink):
                critical_paths.append(path)

    return critical_paths

def print_cpm_table(cpm_result: List[dict]) -> None:
    """Печатает таблицу расчётов CPM"""

    headers = [
        "Код",
        "Работа",
        "Длит.",
        "Предш.",
        "ES",
        "EF",
        "LS",
        "LF",
        "Резерв",
        "Крит.",
    ]

    rows = []
    for row in cpm_result:
        rows.append(
            [
                row["code"],
                row["name"],
                str(row["duration"]),
                ", ".join(row["predecessors"]) or "—",
                str(row["ES"]),
                str(row["EF"]),
                str(row["LS"]),
                str(row["LF"]),
                str(row["float"]),
                "Да" if row["is_critical"] else "Нет",
            ]
        )

    column_widths = []
    for column_index in range(len(headers)):
        max_width = len(headers[column_index])
        for row in rows:
            max_width = max(max_width, len(row[column_index]))
        column_widths.append(max_width)

    def format_row(values: List[str]) -> str:
        return " | ".join(
            value.ljust(column_widths[index])
            for index, value in enumerate(values)
        )

    print(format_row(headers))
    print("-" * (sum(column_widths) + 3 * (len(headers) - 1)))

    for row in rows:
        print(format_row(row))

def draw_network_graph(
    graph: nx.DiGraph,
    activities: Dict[str, Activity],
    cpm_result: List[dict],
    output_path: str | Path = "network_graph.png",
) -> None:
    """Строит и сохраняет сетевой график проекта."""

    critical_nodes = {
        row["code"]
        for row in cpm_result
        if row["is_critical"]
    }

    node_colors = ["tomato" if node in critical_nodes else "lightblue" for node in graph.nodes]

    edge_colors = []
    edge_widths = []

    for source, target in graph.edges:
        if source in critical_nodes and target in critical_nodes:
            edge_colors.append("red")
            edge_widths.append(2.5)
        else:
            edge_colors.append("gray")
            edge_widths.append(1.2)

    labels = {
        node: f"{node}\n{activities[node].duration} дн."
        for node in graph.nodes
    }

    position = nx.spring_layout(graph, seed=7, k=1.4)

    plt.figure(figsize=(15, 9))

    nx.draw_networkx_nodes(
        graph,
        position,
        node_color=node_colors,
        node_size=2600,
        edgecolors="black",
        linewidths=1,
    )

    nx.draw_networkx_edges(
        graph,
        position,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=22,
        connectionstyle="arc3,rad=0.05",
    )

    nx.draw_networkx_labels(
        graph,
        position,
        labels=labels,
        font_size=9,
        font_weight="bold",
    )

    plt.title(
        "Сетевой график проекта цифровизации образовательного учреждения",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def draw_gantt_chart(
    cpm_result: List[dict],
    output_path: str | Path = "gantt_chart.png",
) -> None:
    """Строит диаграмму Ганта по ранним срокам начала работ."""

    sorted_rows = sorted(cpm_result, key=lambda row: row["ES"])

    codes = [row["code"] for row in sorted_rows]
    starts = [row["ES"] for row in sorted_rows]
    durations = [row["duration"] for row in sorted_rows]
    colors = ["tomato" if row["is_critical"] else "lightblue" for row in sorted_rows]

    plt.figure(figsize=(14, 8))

    plt.barh(
        y=codes,
        width=durations,
        left=starts,
        color=colors,
        edgecolor="black",
    )

    for row in sorted_rows:
        plt.text(
            row["ES"] + row["duration"] / 2,
            row["code"],
            f"{row['code']}: {row['duration']} дн.",
            ha="center",
            va="center",
            fontsize=8,
        )

    plt.xlabel("Дни проекта")
    plt.ylabel("Работы")
    plt.title("Календарное представление работ проекта по ранним срокам")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def main() -> None:
    data_file = "activities.json"

    activities = load_activities_from_json(data_file)
    graph = build_graph(activities)
    cpm_result, project_duration = calculate_cpm(graph, activities)
    critical_paths = find_critical_paths(graph, cpm_result)

    print("\nИСХОДНЫЙ ФАЙЛ ДАННЫХ:")
    print(data_file)

    print("\nТАБЛИЦА РАСЧЁТА CPM")
    print_cpm_table(cpm_result)

    print("\nОБЩАЯ ДЛИТЕЛЬНОСТЬ ПРОЕКТА:")
    print(f"{project_duration} дней")

    print("\nКРИТИЧЕСКИЙ ПУТЬ:")
    for number, path in enumerate(critical_paths, start=1):
        print(f"{number}. {' -> '.join(path)}")

    print("\nКРИТИЧЕСКИЕ РАБОТЫ:")
    for row in cpm_result:
        if row["is_critical"]:
            print(f"{row['code']} — {row['name']}")

    draw_network_graph(graph, activities, cpm_result, "network_graph.png")
    draw_gantt_chart(cpm_result, "gantt_chart.png")

    print("\nФайлы сохранены:")
    print("1. network_graph.png — сетевой график")
    print("2. gantt_chart.png — диаграмма Ганта")


if __name__ == "__main__":
    main()
