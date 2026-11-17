from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True

from main import FieldGeometry, Machine, Plan, evaluate_strategies, plan_is_better


STRATEGIES = (
    ("1_Пополам", "strat1"),
    ("2_Полосами", "strat2"),
    ("3_По_производительности", "strat3"),
)

RESULT_COLUMNS = tuple(
    column
    for _, suffix in STRATEGIES
    for column in (
        f"T_{suffix}",
        f"turns_{suffix}",
        f"delay_{suffix}",
        f"status_{suffix}",
    )
) + ("Winner", "Winner_time")

REQUIRED_COLUMNS = {
    "s",
    "l1",
    "l2",
    "U1",
    "U2",
    "hx1",
    "hy1",
    "hx2",
    "hy2",
    "coord1_x",
    "coord1_y",
    "coord2_x",
    "coord2_y",
    "orient1",
    "orient2",
}


def read_xlsx(input_path: Path) -> pd.DataFrame:
    """Прочитать входную таблицу XLSX любого размера."""

    if input_path.suffix.lower() != ".xlsx":
        raise ValueError("Входной файл должен иметь расширение .xlsx")
    return pd.read_excel(input_path)  # pyright: ignore[reportUnknownMemberType]


def save_xlsx(frame: pd.DataFrame, output_path: Path) -> Path:
    """Сохранить результаты проверки в XLSX."""

    if output_path.suffix.lower() != ".xlsx":
        raise ValueError("Выходной файл должен иметь расширение .xlsx")
    frame.to_excel(output_path, index=False)  # pyright: ignore[reportUnknownMemberType]
    return output_path


def number(row: pd.Series, name: str, *, positive: bool = False) -> float:
    """Получить из строки конечное числовое значение."""

    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"поле {name!r} должно быть числом") from error
    if not math.isfinite(value):
        raise ValueError(f"поле {name!r} должно быть конечным числом")
    if positive and value <= 0.0:
        raise ValueError(f"поле {name!r} должно быть больше нуля")
    return value


def optional_number(row: pd.Series, name: str, default: float,
                    *, positive: bool = False) -> float:
    """Прочитать необязательное число или использовать значение по умолчанию."""

    if name not in row or pd.isna(row[name]):
        return default
    return number(row, name, positive=positive)


def machine_from_row(row: pd.Series, index: int) -> Machine:
    """Восстановить описание одной машины из строки входной таблицы."""

    header = number(row, f"l{index}", positive=True)
    work_speed = number(row, f"U{index}", positive=True)
    length = number(row, f"hx{index}", positive=True)
    width = number(row, f"hy{index}", positive=True)
    travel_speed = optional_number(
        row, f"Vtravel{index}", work_speed * 1.25, positive=True
    )
    turn_speed = optional_number(
        row, f"Vturn{index}", work_speed * 0.75, positive=True
    )
    turn_radius = optional_number(
        row, f"R{index}", max(2.5, length / 2.0, width / 2.0), positive=True
    )

    return Machine(
        header=header,
        work_speed=work_speed,
        travel_speed=travel_speed,
        turn_speed=turn_speed,
        length=length,
        width=width,
        turn_radius=turn_radius,
        start=(number(row, f"coord{index}_x"), number(row, f"coord{index}_y")),
        orientation=math.radians(number(row, f"orient{index}")),
    )


def plan_value(plan: Plan, attribute: str, digits: int = 1) -> float | None:
    """Получить округлённый показатель допустимого плана."""

    if not plan.valid:
        return None
    raw_value = getattr(plan, attribute)
    if not isinstance(raw_value, (int, float)):
        raise TypeError(f"Атрибут {attribute} не является числом")
    return round(float(raw_value), digits)


def check_scenario(row: pd.Series) -> dict[str, object]:
    """Рассчитать стратегии и проверить безопасность одного сценария."""

    axis = str(row["s"]).strip().lower()
    if axis not in {"x", "y"}:
        raise ValueError("поле 's' должно содержать 'x' или 'y'")

    machines = (machine_from_row(row, 1), machine_from_row(row, 2))
    plans = evaluate_strategies(FieldGeometry(axis), machines)
    winner: Plan | None = None
    for plan in plans.values():
        if plan.valid and plan_is_better(plan, winner):
            winner = plan

    result: dict[str, object] = {}
    for plan_name, suffix in STRATEGIES:
        plan = plans[plan_name]
        result[f"T_{suffix}"] = plan_value(plan, "makespan")
        result[f"turns_{suffix}"] = plan_value(plan, "turns", 0)
        result[f"delay_{suffix}"] = (
            round(max(plan.delay_1, plan.delay_2)) if plan.valid else None
        )
        result[f"status_{suffix}"] = plan.status

    result["Winner"] = winner.name if winner else "Нет допустимой стратегии"
    result["Winner_time"] = round(winner.makespan, 1) if winner else None
    return result


def check_scenarios(frame: pd.DataFrame, *,
                    show_progress: bool = False) -> pd.DataFrame:
    """Проверить все строки входной таблицы независимо от их количества."""

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Во входной таблице отсутствуют столбцы: {', '.join(missing)}")
    rows: list[dict[str, object]] = []
    total = len(frame)
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        scenario_id = row.get("scenario_id", position)
        try:
            result = check_scenario(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Ошибка в сценарии {scenario_id}: {error}") from error
        row_data: dict[str, object] = {}
        for column, value in row.items():
            row_data[str(column)] = value
        row_data.update(result)
        rows.append(row_data)
        if show_progress and (position == 1 or position % 10 == 0 or position == total):
            print(f"Проверено сценариев: {position}/{total}")

    if rows:
        return pd.DataFrame(rows)

    empty_result = frame.copy()
    for column in RESULT_COLUMNS:
        empty_result[column] = None
    return empty_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверить маршруты и стратегии для сгенерированных сценариев."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("synthetic_scenarios.xlsx"),
        help="входной файл сценариев .xlsx",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("synthetic_data_experiment_v2.xlsx"),
        help="выходной файл результатов .xlsx",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = read_xlsx(args.input)
    results = check_scenarios(scenarios, show_progress=True)
    output_path = save_xlsx(results, args.output)

    print(f"Результат проверки сохранён: {output_path}")
    if results.empty:
        print("Во входной таблице нет сценариев для проверки.")
        return
    print("\nПобедители:")
    print(results["Winner"].value_counts(dropna=False))
    print("\nНедопустимые стратегии:")
    for _, suffix in STRATEGIES:
        column = f"status_{suffix}"
        print(column, int((results[column] != "OK").sum()))


if __name__ == "__main__":
    main()
