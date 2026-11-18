"""Формирование комплекта цифровой проверки по обязательным сценариям А–Г."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from generate import generate_required_scenarios, save_xlsx
from main import ScenarioEvaluation, check_scenarios, evaluate_scenario, save_private_report
from planner import Plan, plan_is_better


def best_improved(evaluation: ScenarioEvaluation) -> Plan | None:
    best: Plan | None = None
    for name in ("2_Полосами", "3_По_производительности"):
        plan = evaluation.plans[name]
        if plan_is_better(plan, best):
            best = plan
    return best


def comparison_rows(
    scenarios: pd.DataFrame,
    evaluations: list[ScenarioEvaluation],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (_, source), evaluation in zip(scenarios.iterrows(), evaluations, strict=True):
        baseline = evaluation.plans["1_Пополам"]
        improved = best_improved(evaluation)
        delta = None
        percent = None
        if baseline.valid and improved is not None and improved.valid:
            delta = baseline.makespan - improved.makespan
            percent = 100.0 * delta / baseline.makespan
        rows.append(
            {
                "scenario_id": source["scenario_id"],
                "description": source["description"],
                "baseline_status": baseline.status,
                "baseline_axis": baseline.axis.upper(),
                "baseline_time": round(baseline.makespan, 1) if baseline.valid else None,
                "improved_strategy": improved.name if improved else None,
                "improved_status": improved.status if improved else "нет допустимого плана",
                "improved_axis": improved.axis.upper() if improved else None,
                "improved_time": round(improved.makespan, 1) if improved else None,
                "delta_seconds": round(delta, 1) if delta is not None else None,
                "delta_percent": round(percent, 2) if percent is not None else None,
                "winner": evaluation.result["Winner"],
                "winner_axis": evaluation.result["Winner_axis"],
            }
        )
    return rows


def comparison_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Сравнение базового и улучшенного алгоритмов",
        "",
        "Базовый алгоритм — равное деление при направлении, заданном в конфигурации. ",
        "Улучшенный алгоритм выбирает между полосным и производительным делением ",
        "и автоматически сравнивает направления X и Y.",
        "",
        "| Сценарий | Базовый план | Улучшенный план | Изменение времени | Победитель |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        baseline = (
            f"{row['baseline_time']} с, {row['baseline_axis']}"
            if row["baseline_time"] is not None
            else str(row["baseline_status"])
        )
        improved = (
            f"{row['improved_strategy']}, {row['improved_time']} с, "
            f"{row['improved_axis']}"
            if row["improved_time"] is not None
            else str(row["improved_status"])
        )
        change = (
            f"{row['delta_seconds']:+} с ({row['delta_percent']:+}%)"
            if row["delta_seconds"] is not None
            else "—"
        )
        lines.append(
            f"| {row['scenario_id']} — {row['description']} | {baseline} | "
            f"{improved} | {change} | {row['winner']} ({row['winner_axis']}) |"
        )
    lines.extend(
        [
            "",
            "Положительное изменение означает, что улучшенный вариант завершает работу ",
            "быстрее базового. Отрицательное значение означает, что базовое равное ",
            "деление оказалось лучше для данного набора ограничений.",
            "",
            "Все варианты используют одинаковую целевую площадь, параметры машин, ",
            "допуск покрытия и правила безопасности.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_required_deliverables(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = generate_required_scenarios()
    save_xlsx(scenarios, output_dir / "required_scenarios.xlsx")
    summary = check_scenarios(scenarios)
    save_xlsx(summary, output_dir / "required_scenarios_results.xlsx")

    evaluations: list[ScenarioEvaluation] = []
    for _, row in scenarios.iterrows():
        evaluation = evaluate_scenario(row)
        evaluations.append(evaluation)
        scenario_id = str(row["scenario_id"])
        save_private_report(
            row,
            evaluation,
            output_dir / f"scenario_{scenario_id}_report.xlsx",
        )
    comparison = comparison_rows(scenarios, evaluations)
    pd.DataFrame(comparison).to_excel(
        output_dir / "baseline_vs_improved.xlsx",
        index=False,
    )
    (output_dir / "BASELINE_VS_IMPROVED.md").write_text(
        comparison_markdown(comparison),
        encoding="utf-8",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сформировать комплект цифровой проверки сценариев А–Г."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("deliverables"),
        help="каталог итогового комплекта",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_required_deliverables(args.output_dir)
    print(f"Комплект проверки сохранён: {output.resolve()}")


if __name__ == "__main__":
    main()


