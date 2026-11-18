from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generate import generate_required_scenarios, generate_scenarios, save_xlsx
from main import check_scenarios, evaluate_scenario, read_xlsx


class IntegrationTests(unittest.TestCase):
    def test_synthetic_matrix_has_162_rows(self) -> None:
        frame = generate_scenarios()
        self.assertEqual(len(frame), 162)
        self.assertEqual(set(frame["s"]), {"x", "y"})
        self.assertIn("allowed_gap_percent", frame.columns)

    def test_required_scenarios_match_assignment(self) -> None:
        frame = generate_required_scenarios()
        self.assertEqual(list(frame["scenario_id"]), ["А", "Б", "В", "Г"])
        sizes = list(zip(frame["field_x"], frame["field_y"], strict=True))
        self.assertEqual(sizes, [(240.0, 120.0), (120.0, 240.0), (240.0, 120.0), (250.0, 130.0)])
        scenario_v = frame.loc[frame["scenario_id"] == "В"].iloc[0]
        self.assertLess(scenario_v["U2"], scenario_v["U1"])

    def test_batch_mode_can_compare_improved_axes(self) -> None:
        row = generate_required_scenarios().iloc[0]
        evaluation = evaluate_scenario(row)
        self.assertEqual(evaluation.plans["1_Пополам"].axis, "x")
        self.assertIn(evaluation.plans["2_Полосами"].axis, {"x", "y"})
        self.assertIn(
            evaluation.plans["3_По_производительности"].axis,
            {"x", "y"},
        )
        self.assertIn(evaluation.result["Winner_axis"], {"X", "Y"})

    def test_private_mode_applies_selected_axis_to_every_strategy(self) -> None:
        row = generate_required_scenarios().iloc[0]
        evaluation_x = evaluate_scenario(row, compare_improved_axes=False)
        row_y = row.copy()
        row_y["s"] = "y"
        evaluation_y = evaluate_scenario(row_y, compare_improved_axes=False)

        self.assertEqual(
            {plan.axis for plan in evaluation_x.plans.values()},
            {"x"},
        )
        self.assertEqual(
            {plan.axis for plan in evaluation_y.plans.values()},
            {"y"},
        )
        self.assertEqual(evaluation_x.result["Winner_axis"], "X")
        self.assertEqual(evaluation_y.result["Winner_axis"], "Y")
        self.assertNotEqual(
            evaluation_x.result["Winner_time"],
            evaluation_y.result["Winner_time"],
        )

    def test_xlsx_roundtrip_and_batch_result(self) -> None:
        frame = generate_required_scenarios().head(1)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.xlsx"
            save_xlsx(frame, source)
            loaded = read_xlsx(source)
            result = check_scenarios(loaded)
        self.assertEqual(len(result), 1)
        self.assertIn("Winner", result.columns)
        self.assertIn("work_strat1", result.columns)
        self.assertIn("coverage_strat1", result.columns)
        self.assertEqual(result.iloc[0]["coverage_strat1"], 100.0)


if __name__ == "__main__":
    unittest.main()
