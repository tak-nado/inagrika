from __future__ import annotations

import unittest

from planner import (
    FieldGeometry,
    Machine,
    Segment,
    coverage_metrics,
    build_route,
    evaluate_assignment,
    evaluate_strategies,
    perimeter_return_path,
    region_passes,
    segments_collide,
)


def machine(start: tuple[float, float]) -> Machine:
    return Machine(
        header=10.0,
        work_speed=2.0,
        travel_speed=2.5,
        turn_speed=1.5,
        length=4.0,
        width=3.0,
        turn_radius=2.5,
        start=start,
        orientation=0.0,
    )


class PlannerTests(unittest.TestCase):
    def test_region_passes_cover_region_without_gaps(self) -> None:
        self.assertEqual(region_passes(0.0, 40.0, 10.0), [5.0, 15.0, 25.0, 35.0])

    def test_coverage_metrics_detect_complete_coverage(self) -> None:
        geometry = FieldGeometry("x", 100.0, 40.0, 8.0)
        machines = (machine((0.0, 5.0)), machine((0.0, 35.0)))
        overlap, uncovered, coverage = coverage_metrics(
            geometry,
            machines,
            ([5.0, 15.0], [25.0, 35.0]),
        )
        self.assertAlmostEqual(overlap, 0.0)
        self.assertAlmostEqual(uncovered, 0.0)
        self.assertAlmostEqual(coverage, 100.0)

    def test_coverage_metrics_detect_gap(self) -> None:
        geometry = FieldGeometry("x", 100.0, 40.0, 8.0)
        machines = (machine((0.0, 5.0)), machine((0.0, 35.0)))
        _, uncovered, coverage = coverage_metrics(
            geometry,
            machines,
            ([5.0], [35.0]),
        )
        self.assertGreater(uncovered, 0.0)
        self.assertAlmostEqual(coverage, 50.0)

    def test_temporal_crossing_is_checked_by_time(self) -> None:
        first = Segment((0.0, 0.0), (10.0, 0.0), 0.0, 10.0, "work")
        simultaneous = Segment((5.0, -5.0), (5.0, 5.0), 0.0, 10.0, "work")
        later = Segment((5.0, -5.0), (5.0, 5.0), 20.0, 30.0, "work")
        self.assertTrue(segments_collide(first, simultaneous, 1.0))
        self.assertFalse(segments_collide(first, later, 1.0))

    def test_three_strategies_are_returned(self) -> None:
        geometry = FieldGeometry("x", 100.0, 40.0, 12.0)
        machines = (machine((0.0, 5.0)), machine((0.0, 35.0)))
        plans = evaluate_strategies(geometry, machines)
        self.assertEqual(
            set(plans),
            {"1_Пополам", "2_Полосами", "3_По_производительности"},
        )
        self.assertTrue(any(plan.valid for plan in plans.values()))
        for plan in plans.values():
            if plan.valid:
                self.assertEqual(plan.axis, "x")
                self.assertAlmostEqual(plan.coverage_percent, 100.0)

    def test_gap_tolerance_rejects_incomplete_coverage(self) -> None:
        geometry = FieldGeometry("x", 100.0, 40.0, 12.0, 0.0)
        machines = (machine((0.0, 5.0)), machine((0.0, 35.0)))
        plan = evaluate_assignment("gap", geometry, machines, ([5.0], [35.0]))
        self.assertFalse(plan.valid)
        self.assertIn("пропуски", plan.status)
        self.assertAlmostEqual(plan.coverage_percent, 50.0)

    def test_narrow_headland_rejects_route(self) -> None:
        geometry = FieldGeometry("x", 100.0, 40.0, 2.0)
        route = build_route(machine((0.0, 5.0)), geometry, [5.0, 15.0], False, False)
        self.assertIsNone(route)

    def test_optional_return_to_start_follows_perimeter(self) -> None:
        geometry = FieldGeometry("x", 100.0, 40.0, 12.0, 0.0, True)
        source = machine((0.0, 5.0))
        path = perimeter_return_path((100.0, 15.0), source.start, geometry)
        self.assertEqual(path[-1], source.start)
        route = build_route(source, geometry, [5.0, 15.0], False, False)
        self.assertIsNotNone(route)
        if route is not None:
            self.assertEqual(route.finish, source.start)


if __name__ == "__main__":
    unittest.main()
