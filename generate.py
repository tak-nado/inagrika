from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True

from main import FIELD_X, FIELD_Y, make_machine


BASE_HEADER = 6.0
BASE_WORK_SPEED = 2.0
BASE_LENGTH = 5.0
BASE_WIDTH = 3.0

HEADER_VARIANTS = {"=": BASE_HEADER, ">": 8.0, "<": 4.5}
SPEED_VARIANTS = {"=": BASE_WORK_SPEED, ">": 3.0, "<": 1.5}
LENGTH_VARIANTS = {"=": BASE_LENGTH, ">": 6.0, "<": 4.0}
WIDTH_VARIANTS = {"=": BASE_WIDTH, ">": 3.5, "<": 2.5}


def condition_values(condition: tuple[str, str, str, str]) -> tuple[float, ...]:
    """Преобразовать отношения параметров в значения для двух машин."""

    header_condition, speed_condition, length_condition, width_condition = condition

    def pair(values: dict[str, float], relation: str) -> tuple[float, float]:
        return (values[relation], values["="])

    header_1, header_2 = pair(HEADER_VARIANTS, header_condition)
    speed_1, speed_2 = pair(SPEED_VARIANTS, speed_condition)
    length_1, length_2 = pair(LENGTH_VARIANTS, length_condition)
    width_1, width_2 = pair(WIDTH_VARIANTS, width_condition)
    return header_1, header_2, speed_1, speed_2, length_1, length_2, width_1, width_2


def generate_scenarios() -> pd.DataFrame:
    """Сформировать входную таблицу из 162 сценариев без их проверки."""

    rows: list[dict[str, object]] = []
    relations: tuple[str, str, str] = ("=", ">", "<")
    conditions = [
        (header, speed, length, width)
        for header in relations
        for speed in relations
        for length in relations
        for width in relations
    ]

    for axis in ("x", "y"):
        for condition in conditions:
            header_relation, speed_relation, length_relation, width_relation = condition
            (header_1, header_2, speed_1, speed_2,
             length_1, length_2, width_1, width_2) = condition_values(condition)

            if axis == "x":
                start_1 = (0.0, header_1 / 2.0)
                start_2 = (0.0, FIELD_Y - header_2 / 2.0)
                orientation_1 = orientation_2 = 0.0
                sweep_width = FIELD_Y
            else:
                start_1 = (header_1 / 2.0, 0.0)
                start_2 = (FIELD_X - header_2 / 2.0, 0.0)
                orientation_1 = orientation_2 = 90.0
                sweep_width = FIELD_X

            machine_1 = make_machine(
                header_1, speed_1, length_1, width_1, start_1, orientation_1
            )
            machine_2 = make_machine(
                header_2, speed_2, length_2, width_2, start_2, orientation_2
            )
            gap_tolerance = max(
                (sweep_width % header_1) / sweep_width,
                (sweep_width % header_2) / sweep_width,
            )

            rows.append(
                {
                    "scenario_id": len(rows) + 1,
                    "s": axis,
                    "cond_l": header_relation,
                    "cond_U": speed_relation,
                    "cond_hx": length_relation,
                    "cond_hy": width_relation,
                    "l1": header_1,
                    "l2": header_2,
                    "U1": speed_1,
                    "U2": speed_2,
                    "Vtravel1": machine_1.travel_speed,
                    "Vtravel2": machine_2.travel_speed,
                    "Vturn1": machine_1.turn_speed,
                    "Vturn2": machine_2.turn_speed,
                    "hx1": length_1,
                    "hy1": width_1,
                    "hx2": length_2,
                    "hy2": width_2,
                    "R1": machine_1.turn_radius,
                    "R2": machine_2.turn_radius,
                    "coord1_x": start_1[0],
                    "coord1_y": start_1[1],
                    "coord2_x": start_2[0],
                    "coord2_y": start_2[1],
                    "orient1": orientation_1,
                    "orient2": orientation_2,
                    "propusk_%": round(gap_tolerance * 100.0, 2),
                }
            )

    return pd.DataFrame(rows)


def save_xlsx(frame: pd.DataFrame, output_path: Path) -> Path:
    """Сохранить таблицу в XLSX и вернуть её путь."""

    if output_path.suffix.lower() != ".xlsx":
        raise ValueError("Выходной файл должен иметь расширение .xlsx")
    frame.to_excel(output_path, index=False)  # pyright: ignore[reportUnknownMemberType]
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сформировать входные сценарии для проверки стратегий уборки."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("synthetic_scenarios.xlsx"),
        help="выходной файл .xlsx",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = generate_scenarios()
    output_path = save_xlsx(scenarios, args.output)
    print(f"Сформировано сценариев: {len(scenarios)}")
    print(f"Исходные данные сохранены: {output_path}")


if __name__ == "__main__":
    main()
