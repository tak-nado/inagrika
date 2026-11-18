from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

sys.dont_write_bytecode = True

from PyQt6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QThread,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from planner import (
    FIELD_X,
    FIELD_Y,
    TURN_ZONE_WIDTH,
    FieldGeometry,
    Machine,
    Plan,
    Route,
    evaluate_strategies,
    plan_is_better,
)


STRATEGIES = (
    ("1_Пополам", "strat1", "Пополам"),
    ("2_Полосами", "strat2", "Полосами"),
    ("3_По_производительности", "strat3", "По производительности"),
)

REQUIRED_COLUMNS = {
    "s", "l1", "l2", "U1", "U2", "hx1", "hy1", "hx2", "hy2",
    "coord1_x", "coord1_y", "coord2_x", "coord2_y", "orient1", "orient2",
}

RESULT_COLUMNS = tuple(
    column
    for _, suffix, _ in STRATEGIES
    for column in (
        f"axis_{suffix}", f"T_{suffix}", f"T1_{suffix}", f"T2_{suffix}",
        f"work_{suffix}", f"deadhead_{suffix}", f"overlap_{suffix}",
        f"uncovered_{suffix}", f"coverage_{suffix}", f"turns_{suffix}",
        f"delay_{suffix}", f"status_{suffix}",
    )
) + ("Winner", "Winner_axis", "Winner_time")


def strategy_display_name(plan_name: str) -> str:
    for name, _, label in STRATEGIES:
        if name == plan_name:
            return label
    return plan_name.replace("_", " ")


@dataclass
class ScenarioEvaluation:
    result: dict[str, object]
    plans: dict[str, Plan]
    winner: Plan | None
    geometry: FieldGeometry


def read_xlsx(input_path: Path) -> pd.DataFrame:
    """Прочитать XLSX с произвольным количеством сценариев."""

    if input_path.suffix.lower() != ".xlsx":
        raise ValueError("Поддерживаются только входные файлы .xlsx")
    return pd.read_excel(input_path)  # pyright: ignore[reportUnknownMemberType]


def save_xlsx(frame: pd.DataFrame, output_path: Path) -> Path:
    """Сохранить таблицу результатов в XLSX."""

    if output_path.suffix.lower() != ".xlsx":
        raise ValueError("Результат должен иметь расширение .xlsx")
    frame.to_excel(output_path, index=False)  # pyright: ignore[reportUnknownMemberType]
    return output_path


def number(row: pd.Series, name: str, *, positive: bool = False) -> float:
    """Получить из строки конечное числовое значение."""

    try:
        value = float(cast(Any, row[name]))
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

    if name not in row:
        return default
    raw_value = cast(Any, row[name])
    if cast(bool, pd.isna(raw_value)):
        return default
    return number(row, name, positive=positive)


def optional_bool(row: pd.Series, name: str, default: bool = False) -> bool:
    """Прочитать необязательный логический параметр из XLSX или формы."""

    if name not in row:
        return default
    raw_value = cast(Any, row[name])
    if cast(bool, pd.isna(raw_value)):
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        if float(raw_value) in {0.0, 1.0}:
            return bool(raw_value)
    text_value = str(raw_value).strip().lower()
    if text_value in {"1", "true", "yes", "да"}:
        return True
    if text_value in {"0", "false", "no", "нет"}:
        return False
    raise ValueError(f"поле {name!r} должно содержать да/нет или 1/0")


def comparison_sign(first: float, second: float) -> str:
    """Вернуть знак отношения фактических числовых значений."""

    if math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-9):
        return "="
    return ">" if first > second else "<"


def geometry_from_row(row: pd.Series) -> FieldGeometry:
    """Проверить и получить геометрию поля."""

    axis = str(row["s"]).strip().lower()
    if axis not in {"x", "y"}:
        raise ValueError("поле 's' должно содержать 'x' или 'y'")
    field_x = optional_number(row, "field_x", FIELD_X, positive=True)
    field_y = optional_number(row, "field_y", FIELD_Y, positive=True)
    turn_zone = optional_number(
        row, "turn_zone_width", TURN_ZONE_WIDTH, positive=True
    )
    allowed_gap_percent = optional_number(row, "allowed_gap_percent", 0.0)
    return_to_start = optional_bool(row, "return_to_start", False)
    if not 0.0 <= allowed_gap_percent <= 100.0:
        raise ValueError("поле 'allowed_gap_percent' должно быть от 0 до 100")
    if 2.0 * turn_zone >= min(field_x, field_y):
        raise ValueError("две разворотные зоны должны быть уже каждой стороны поля")
    return FieldGeometry(
        axis,
        field_x,
        field_y,
        turn_zone,
        allowed_gap_percent / 100.0,
        return_to_start,
    )


def machine_from_row(row: pd.Series, index: int,
                     geometry: FieldGeometry) -> Machine:
    """Восстановить и проверить описание машины из числовых полей."""

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
    start_x = number(row, f"coord{index}_x")
    start_y = number(row, f"coord{index}_y")
    if not 0.0 <= start_x <= geometry.field_x:
        raise ValueError(f"coord{index}_x должна находиться в пределах поля")
    if not 0.0 <= start_y <= geometry.field_y:
        raise ValueError(f"coord{index}_y должна находиться в пределах поля")
    return Machine(
        header=header,
        work_speed=work_speed,
        travel_speed=travel_speed,
        turn_speed=turn_speed,
        length=length,
        width=width,
        turn_radius=turn_radius,
        start=(start_x, start_y),
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


def evaluate_scenario(
    row: pd.Series,
    *,
    compare_improved_axes: bool = True,
) -> ScenarioEvaluation:
    """Рассчитать планы с фиксированным или оптимизируемым направлением."""

    selected_geometry = geometry_from_row(row)
    machines = (
        machine_from_row(row, 1, selected_geometry),
        machine_from_row(row, 2, selected_geometry),
    )
    axes = ("x", "y") if compare_improved_axes else (selected_geometry.axis,)
    geometries = {
        axis: FieldGeometry(
            axis,
            selected_geometry.field_x,
            selected_geometry.field_y,
            selected_geometry.turn_zone_width,
            selected_geometry.allowed_gap_fraction,
            selected_geometry.return_to_start,
        )
        for axis in axes
    }
    evaluated = {
        axis: evaluate_strategies(geometry, machines)
        for axis, geometry in geometries.items()
    }
    plans: dict[str, Plan] = {
        "1_Пополам": evaluated[selected_geometry.axis]["1_Пополам"]
    }
    for plan_name in ("2_Полосами", "3_По_производительности"):
        best_plan: Plan | None = None
        for axis in axes:
            candidate = evaluated[axis][plan_name]
            if plan_is_better(candidate, best_plan):
                best_plan = candidate
        plans[plan_name] = best_plan or evaluated[selected_geometry.axis][plan_name]

    winner: Plan | None = None
    for plan in plans.values():
        if plan.valid and plan_is_better(plan, winner):
            winner = plan

    result: dict[str, object] = {
        "cond_l": comparison_sign(machines[0].header, machines[1].header),
        "cond_U": comparison_sign(machines[0].work_speed, machines[1].work_speed),
        "cond_hx": comparison_sign(machines[0].length, machines[1].length),
        "cond_hy": comparison_sign(machines[0].width, machines[1].width),
    }
    for plan_name, suffix, _ in STRATEGIES:
        plan = plans[plan_name]
        result[f"axis_{suffix}"] = plan.axis.upper() if plan.axis else None
        result[f"T_{suffix}"] = plan_value(plan, "makespan")
        result[f"T1_{suffix}"] = plan_value(plan, "time_1")
        result[f"T2_{suffix}"] = plan_value(plan, "time_2")
        result[f"work_{suffix}"] = plan_value(plan, "work_distance")
        result[f"deadhead_{suffix}"] = plan_value(plan, "deadhead")
        result[f"overlap_{suffix}"] = plan_value(plan, "overlap_area")
        result[f"uncovered_{suffix}"] = plan_value(plan, "uncovered_area")
        result[f"coverage_{suffix}"] = plan_value(plan, "coverage_percent", 2)
        result[f"turns_{suffix}"] = plan_value(plan, "turns", 0)
        result[f"delay_{suffix}"] = (
            round(max(plan.delay_1, plan.delay_2)) if plan.valid else None
        )
        result[f"status_{suffix}"] = plan.status
    result["Winner"] = winner.name if winner else "Нет допустимой стратегии"
    result["Winner_time"] = round(winner.makespan, 1) if winner else None
    result["Winner_axis"] = winner.axis.upper() if winner else None
    winner_geometry = geometries[winner.axis] if winner else selected_geometry
    return ScenarioEvaluation(result, plans, winner, winner_geometry)


def check_scenario(row: pd.Series) -> dict[str, object]:
    """Вернуть табличный результат проверки одного сценария."""

    return evaluate_scenario(row).result


def check_scenarios(
    frame: pd.DataFrame,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Проверить все строки XLSX без ограничения количества сценариев."""

    missing = sorted(REQUIRED_COLUMNS - {str(column) for column in frame.columns})
    if missing:
        raise ValueError(
            f"Во входной таблице отсутствуют столбцы: {', '.join(missing)}"
        )
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
        if progress_callback is not None:
            progress_callback(position, total)
    if rows:
        return pd.DataFrame(rows)
    empty_result = frame.copy()
    for column in ("cond_l", "cond_U", "cond_hx", "cond_hy", *RESULT_COLUMNS):
        if column not in empty_result:
            empty_result[column] = None
    return empty_result


def route_frame(route: Route | None) -> pd.DataFrame:
    """Преобразовать сегменты маршрута в таблицу для отчёта."""

    columns = (
        "segment", "kind", "begin_x", "begin_y", "end_x", "end_y",
        "begin_time", "end_time", "duration",
    )
    if route is None:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for index, segment in enumerate(route.segments, start=1):
        rows.append(
            {
                "segment": index,
                "kind": segment.kind,
                "begin_x": segment.begin[0],
                "begin_y": segment.begin[1],
                "end_x": segment.end[0],
                "end_y": segment.end[1],
                "begin_time": segment.begin_time,
                "end_time": segment.end_time,
                "duration": segment.end_time - segment.begin_time,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def save_private_report(input_row: pd.Series, evaluation: ScenarioEvaluation,
                        output_path: Path) -> Path:
    """Сохранить конфигурацию, показатели и маршруты лучшего плана в XLSX."""

    if output_path.suffix.lower() != ".xlsx":
        raise ValueError("Отчёт должен иметь расширение .xlsx")
    configuration = {str(key): value for key, value in input_row.items()}
    report_row = dict(configuration)
    report_row.update(evaluation.result)
    winner = evaluation.winner
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([configuration]).to_excel(
            writer, sheet_name="configuration", index=False
        )
        pd.DataFrame([report_row]).to_excel(
            writer, sheet_name="results", index=False
        )
        route_frame(winner.route_1 if winner else None).to_excel(
            writer, sheet_name="route_machine_1", index=False
        )
        route_frame(winner.route_2 if winner else None).to_excel(
            writer, sheet_name="route_machine_2", index=False
        )
    return output_path


class TableWorker(QObject):
    progress = pyqtSignal(int, int, arguments=["done", "total"])
    completed = pyqtSignal(object, str, arguments=["result", "output_path"])
    failed = pyqtSignal(str, arguments=["message"])

    def __init__(self, frame: pd.DataFrame, output_path: Path) -> None:
        super().__init__()
        self.frame = frame
        self.output_path = output_path

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = check_scenarios(
                self.frame,
                progress_callback=lambda done, total: self.progress.emit(done, total),
            )
            save_xlsx(result, self.output_path)
            self.completed.emit(result, str(self.output_path))
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))


class SingleWorker(QObject):
    completed = pyqtSignal(object, arguments=["result"])
    failed = pyqtSignal(str, arguments=["message"])

    def __init__(self, row: pd.Series) -> None:
        super().__init__()
        self.row = row

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.completed.emit(
                evaluate_scenario(self.row, compare_improved_axes=False)
            )
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))


class RouteCanvas(QWidget):
    """Читаемая карта поля, разворотных зон и маршрутов двух машин."""

    MACHINE_1_COLOR = QColor("#e11d48")
    MACHINE_2_COLOR = QColor("#2563eb")

    def __init__(self) -> None:
        super().__init__()
        self.plan: Plan | None = None
        self.field_geometry: FieldGeometry | None = None
        self.setMinimumSize(620, 360)
        self.setObjectName("routeCanvas")

    def set_route(self, plan: Plan | None, geometry: FieldGeometry | None) -> None:
        self.plan = plan
        self.field_geometry = geometry
        self.update()

    def save_png(self, output_path: Path) -> bool:
        """Сохранить схему маршрутов как крупное PNG-изображение."""

        image = QImage(1800, 1100, QImage.Format.Format_ARGB32)
        image.fill(QColor("#f8fafc"))
        painter = QPainter(image)
        self._draw_scene(painter, 1800.0, 1100.0)
        painter.end()
        return image.save(str(output_path), "PNG")

    @staticmethod
    def _route_metrics(route: Route | None) -> tuple[int, int]:
        if route is None:
            return 0, 0
        return route.pass_count, route.turn_count

    def paintEvent(self, event: object) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        self._draw_scene(painter, float(self.width()), float(self.height()))
        painter.end()

    def _draw_scene(self, painter: QPainter,
                    canvas_width: float, canvas_height: float) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        background = QLinearGradient(0.0, 0.0, 0.0, canvas_height)
        background.setColorAt(0.0, QColor("#ffffff"))
        background.setColorAt(1.0, QColor("#f1f5f9"))
        painter.fillRect(QRectF(0.0, 0.0, canvas_width, canvas_height), background)

        title_font = QFont("Segoe UI", 12, QFont.Weight.DemiBold)
        label_font = QFont("Segoe UI", 9)
        small_font = QFont("Segoe UI", 8)
        painter.setFont(title_font)
        painter.setPen(QColor("#0f172a"))
        painter.drawText(24, 31, "Карта выбранного маршрута")

        geometry = self.field_geometry
        if geometry is None:
            panel = QRectF(24.0, 52.0, canvas_width - 48.0, canvas_height - 76.0)
            painter.setPen(QPen(QColor("#cbd5e1"), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QColor("#ffffff"))
            painter.drawRoundedRect(panel, 14.0, 14.0)
            painter.setFont(QFont("Segoe UI", 11))
            painter.setPen(QColor("#64748b"))
            painter.drawText(
                panel,
                Qt.AlignmentFlag.AlignCenter,
                "Введите числовые параметры и нажмите\n«Рассчитать частный случай»",
            )
            return

        plan = self.plan
        subtitle = (
            f"{strategy_display_name(plan.name)}  •  {plan.makespan:.1f} с"
            if plan is not None and plan.valid
            else "допустимый маршрут не найден"
        )
        painter.setFont(label_font)
        painter.setPen(QColor("#475569"))
        painter.drawText(24, 51, subtitle)

        top_margin = 92.0
        left_margin = 76.0
        right_margin = 34.0
        bottom_margin = 68.0
        available_width = max(1.0, canvas_width - left_margin - right_margin)
        available_height = max(1.0, canvas_height - top_margin - bottom_margin)
        scale = min(
            available_width / geometry.field_x,
            available_height / geometry.field_y,
        )
        field_width = geometry.field_x * scale
        field_height = geometry.field_y * scale
        left = left_margin + (available_width - field_width) / 2.0
        top = top_margin + (available_height - field_height) / 2.0
        field_rect = QRectF(left, top, field_width, field_height)

        shadow_rect = field_rect.translated(4.0, 5.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 28))
        painter.drawRoundedRect(shadow_rect, 5.0, 5.0)
        painter.setBrush(QColor("#f0fdf4"))
        painter.setPen(QPen(QColor("#166534"), 2.0))
        painter.drawRect(field_rect)

        # Координатная сетка и подписи позволяют оценить масштаб маршрута.
        painter.setFont(small_font)
        for step in range(1, 10):
            fraction = step / 10.0
            x = left + field_width * fraction
            y = top + field_height * fraction
            painter.setPen(QPen(QColor(148, 163, 184, 85), 1.0))
            painter.drawLine(QPointF(x, top), QPointF(x, top + field_height))
            painter.drawLine(QPointF(left, y), QPointF(left + field_width, y))
        painter.setPen(QColor("#475569"))
        for step in range(0, 11, 2):
            fraction = step / 10.0
            x_value = geometry.field_x * fraction
            y_value = geometry.field_y * fraction
            x = left + field_width * fraction
            y = top + field_height * (1.0 - fraction)
            painter.drawText(QRectF(x - 24.0, top + field_height + 7.0, 48.0, 18.0),
                             Qt.AlignmentFlag.AlignCenter, f"{x_value:g}")
            painter.drawText(QRectF(left - 54.0, y - 9.0, 45.0, 18.0),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{y_value:g}")
        painter.setFont(label_font)
        painter.drawText(QRectF(left, top + field_height + 27.0, field_width, 22.0),
                         Qt.AlignmentFlag.AlignCenter, "X, м")
        painter.save()
        painter.translate(left - 58.0, top + field_height / 2.0)
        painter.rotate(-90.0)
        painter.drawText(QRectF(-50.0, -12.0, 100.0, 24.0),
                         Qt.AlignmentFlag.AlignCenter, "Y, м")
        painter.restore()

        # Разворотные зоны выделены штриховкой, рабочая область остаётся светлой.
        zone = geometry.turn_zone_width * scale
        painter.setBrush(QBrush(QColor(253, 224, 71, 115), Qt.BrushStyle.BDiagPattern))
        painter.setPen(Qt.PenStyle.NoPen)
        if geometry.axis == "x":
            zones = (
                QRectF(left, top, zone, field_height),
                QRectF(left + field_width - zone, top, zone, field_height),
            )
        else:
            zones = (
                QRectF(left, top, field_width, zone),
                QRectF(left, top + field_height - zone, field_width, zone),
            )
        for zone_rect in zones:
            painter.drawRect(zone_rect)

        def point(x: float, y: float) -> QPointF:
            return QPointF(left + x * scale, top + field_height - y * scale)

        def draw_marker(position: tuple[float, float], color: QColor,
                        label: str, finish: bool = False) -> None:
            center = point(*position)
            painter.setPen(QPen(QColor("#ffffff"), 2.0))
            painter.setBrush(color)
            if finish:
                painter.drawRoundedRect(
                    QRectF(center.x() - 5.0, center.y() - 5.0, 10.0, 10.0), 2.0, 2.0
                )
            else:
                painter.drawEllipse(center, 6.0, 6.0)
            painter.setFont(small_font)
            painter.setPen(QColor("#0f172a"))
            label_x = -24.0 if center.x() > left + field_width - 28.0 else 8.0
            label_y = 16.0 if center.y() < top + 16.0 else -7.0
            painter.drawText(center + QPointF(label_x, label_y), label)

        def draw_route(route: Route | None, color: QColor, number: int) -> None:
            if route is None:
                return
            painter.save()
            painter.setClipRect(field_rect.adjusted(-2.0, -2.0, 2.0, 2.0))
            for segment in route.segments:
                style = Qt.PenStyle.SolidLine
                width = 2.8
                visible_color = QColor(color)
                if segment.kind == "deadhead":
                    style, width = Qt.PenStyle.DashLine, 2.1
                    visible_color.setAlpha(205)
                elif segment.kind == "turn":
                    style, width = Qt.PenStyle.DotLine, 2.4
                elif segment.kind in {"wait", "park"}:
                    continue
                painter.setPen(QPen(visible_color, width, style, Qt.PenCapStyle.RoundCap))
                painter.drawLine(point(*segment.begin), point(*segment.end))
            painter.restore()
            draw_marker(route.start, color, f"S{number}")
            draw_marker(route.finish, color, f"F{number}", finish=True)

        if plan is not None and plan.valid:
            draw_route(plan.route_1, self.MACHINE_1_COLOR, 1)
            draw_route(plan.route_2, self.MACHINE_2_COLOR, 2)
        else:
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
            painter.setPen(QColor("#991b1b"))
            painter.drawText(
                field_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Маршрут не построен\nПроверьте причины в таблице стратегий",
            )

        # Легенда использует те же цвета и типы линий, что и карта.
        legend_y = 72.0
        painter.setFont(small_font)
        legend_items = (
            (self.MACHINE_1_COLOR, Qt.PenStyle.SolidLine, "машина 1"),
            (self.MACHINE_2_COLOR, Qt.PenStyle.SolidLine, "машина 2"),
            (QColor("#475569"), Qt.PenStyle.DashLine, "переезд"),
            (QColor("#475569"), Qt.PenStyle.DotLine, "манёвр"),
        )
        legend_x = 24.0
        for color, style, text in legend_items:
            painter.setPen(QPen(color, 2.5, style))
            painter.drawLine(QPointF(legend_x, legend_y), QPointF(legend_x + 24.0, legend_y))
            painter.setPen(QColor("#334155"))
            painter.drawText(QRectF(legend_x + 30.0, legend_y - 9.0, 82.0, 18.0), text)
            legend_x += 112.0
        painter.setPen(QPen(QColor("#ca8a04"), 1.0))
        painter.setBrush(QBrush(QColor("#fde047"), Qt.BrushStyle.BDiagPattern))
        painter.drawRect(QRectF(legend_x, legend_y - 6.0, 22.0, 12.0))
        painter.setPen(QColor("#334155"))
        painter.drawText(
            QRectF(legend_x + 29.0, legend_y - 9.0, 150.0, 18.0),
            "разворотная зона",
        )


class NumericInput(QDoubleSpinBox):
    """Числовое поле без стрелок и случайного изменения колесом мыши."""

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setMinimumWidth(96)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

    def wheelEvent(self, event: QWheelEvent | None) -> None:  # noqa: N802
        if event is not None:
            event.ignore()


def make_spin(value: float, *, minimum: float = 0.001,
              maximum: float = 100000.0, decimals: int = 3) -> QDoubleSpinBox:
    spin = NumericInput()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setSingleStep(0.5)
    spin.setKeyboardTracking(False)
    return spin


def make_page_header(title: str, subtitle: str) -> QFrame:
    frame = QFrame()
    frame.setObjectName("pageHeader")
    frame.setFixedHeight(92)
    title_label = QLabel(title)
    title_label.setObjectName("pageTitle")
    subtitle_label = QLabel(subtitle)
    subtitle_label.setObjectName("pageSubtitle")
    subtitle_label.setWordWrap(True)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 16, 22, 16)
    layout.setSpacing(4)
    layout.addWidget(title_label)
    layout.addWidget(subtitle_label)
    return frame


def load_ui_font() -> str:
    """Явно загрузить шрифт с кириллицей, включая безэкранный режим Qt."""

    font_directory = Path("C:/Windows/Fonts")
    families: list[str] = []
    for filename in ("segoeui.ttf", "seguisb.ttf", "segoeuib.ttf"):
        font_path = font_directory / filename
        if not font_path.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            families.extend(QFontDatabase.applicationFontFamilies(font_id))
    return families[0] if families else "Arial"


def make_card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(10)
    return frame, layout


def make_section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


def show_dialog(parent: QWidget, icon: QMessageBox.Icon,
                title: str, message: str) -> None:
    """Показать светлое диалоговое окно с гарантированно читаемым текстом."""

    dialog = QMessageBox(parent)
    dialog.setIcon(icon)
    dialog.setWindowTitle(title)
    dialog.setText(title)
    dialog.setInformativeText(message)
    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.setStyleSheet(
        """
        QMessageBox { background: #ffffff; }
        QMessageBox QLabel {
            color: #111827; background: transparent;
            font-family: "Segoe UI"; font-size: 11pt;
        }
        QMessageBox QLabel#qt_msgbox_label,
        QMessageBox QLabel#qt_msgbox_informativelabel { min-width: 440px; }
        QMessageBox QPushButton {
            min-width: 100px; min-height: 34px; background: #2563eb;
            color: white; border: none; border-radius: 6px; font-weight: 700;
        }
        QMessageBox QPushButton:hover { background: #1d4ed8; }
        """
    )
    dialog.exec()


class TableTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.frame: pd.DataFrame | None = None
        self.input_path: Path | None = None
        self.output_path: Path | None = None
        self.worker_thread: QThread | None = None
        self.worker: TableWorker | None = None

        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("Выберите исходный XLSX-файл")
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("Путь результата появится автоматически")
        self.rows_label = QLabel("Таблица не загружена")
        self.rows_label.setObjectName("badge")
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.preview = QTableWidget()
        self.preview.setObjectName("dataTable")
        self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview.setAlternatingRowColors(True)
        self.preview.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        preview_header = self.preview.horizontalHeader()
        if preview_header is not None:
            preview_header.setMinimumSectionSize(90)
            preview_header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        preview_vertical_header = self.preview.verticalHeader()
        if preview_vertical_header is not None:
            preview_vertical_header.setDefaultSectionSize(30)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(105)
        self.log.setPlaceholderText("Здесь появится ход проверки")
        self.run_button = QPushButton("Запустить расчёт")
        self.run_button.setEnabled(False)

        input_button = QPushButton("Выбрать XLSX")
        input_button.setProperty("variant", "secondary")
        input_button.clicked.connect(self.choose_input)
        output_button = QPushButton("Куда сохранить")
        output_button.setProperty("variant", "secondary")
        output_button.clicked.connect(self.choose_output)
        self.run_button.clicked.connect(self.start_calculation)

        paths = QGridLayout()
        paths.setHorizontalSpacing(12)
        paths.setVerticalSpacing(10)
        paths.addWidget(QLabel("Исходная таблица:"), 0, 0)
        paths.addWidget(self.input_edit, 0, 1)
        paths.addWidget(input_button, 0, 2)
        paths.addWidget(QLabel("Результат:"), 1, 0)
        paths.addWidget(self.output_edit, 1, 1)
        paths.addWidget(output_button, 1, 2)

        controls = QHBoxLayout()
        controls.addWidget(self.rows_label)
        controls.addStretch()
        controls.addWidget(self.run_button)

        file_card, file_layout = make_card()
        file_layout.addWidget(make_section_title("Файлы расчёта"))
        file_layout.addLayout(paths)
        file_layout.addLayout(controls)
        file_layout.addWidget(self.progress)

        preview_card, preview_layout = make_card()
        preview_layout.addWidget(make_section_title("Данные и результаты"))
        preview_hint = QLabel(
            "Показываются первые 100 строк. Рассчитываются все строки таблицы."
        )
        preview_hint.setObjectName("mutedText")
        preview_layout.addWidget(preview_hint)
        preview_layout.addWidget(self.preview, 1)

        log_card, log_layout = make_card()
        log_layout.addWidget(make_section_title("Состояние"))
        log_layout.addWidget(self.log)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(
            make_page_header(
                "Пакетная проверка сценариев",
                "Загрузите XLSX любого размера. Приложение проверит все строки в фоне "
                "и сохранит отдельный файл с показателями и выбранной стратегией.",
            )
        )
        layout.addWidget(file_card)
        layout.addWidget(preview_card, 1)
        layout.addWidget(log_card)

    def choose_input(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Выберите таблицу сценариев", "", "Excel (*.xlsx)"
        )
        if not filename:
            return
        try:
            path = Path(filename)
            self.frame = read_xlsx(path)
            self.input_path = path
            self.output_path = path.with_name(f"{path.stem}_result.xlsx")
            self.input_edit.setText(str(path))
            self.output_edit.setText(str(self.output_path))
            self.rows_label.setText(f"Сценариев: {len(self.frame)}")
            self.run_button.setEnabled(True)
            self.show_preview(self.frame)
            self.log.setPlainText("Таблица загружена. Все строки будут проверены.")
        except Exception as error:  # noqa: BLE001
            show_dialog(
                self, QMessageBox.Icon.Critical, "Ошибка загрузки", str(error)
            )

    def choose_output(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            str(self.output_path or Path("results.xlsx")),
            "Excel (*.xlsx)",
        )
        if filename:
            path = Path(filename)
            self.output_path = (
                path if path.suffix.lower() == ".xlsx" else path.with_suffix(".xlsx")
            )
            self.output_edit.setText(str(self.output_path))

    def show_preview(self, frame: pd.DataFrame) -> None:
        preview = frame.head(100)
        self.preview.clear()
        self.preview.setRowCount(len(preview))
        self.preview.setColumnCount(len(preview.columns))
        self.preview.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for row_index, (_, row) in enumerate(preview.iterrows()):
            for column_index, value in enumerate(row.tolist()):
                self.preview.setItem(
                    row_index, column_index, QTableWidgetItem(str(value))
                )
        self.preview.resizeColumnsToContents()

    def start_calculation(self) -> None:
        if self.frame is None or self.output_path is None:
            return
        if (
            self.input_path is not None
            and self.input_path.resolve() == self.output_path.resolve()
        ):
            show_dialog(
                self,
                QMessageBox.Icon.Warning,
                "Нельзя перезаписать исходник",
                "Выберите для результата другой XLSX-файл.",
            )
            return
        self.run_button.setEnabled(False)
        self.progress.setValue(0)
        self.log.setPlainText("Расчёт запущен в фоновом потоке…")
        thread = QThread(self)
        worker = TableWorker(self.frame.copy(), self.output_path)
        self.worker_thread = thread
        self.worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.update_progress)
        worker.completed.connect(self.calculation_finished)
        worker.failed.connect(self.calculation_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self.thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def update_progress(self, done: int, total: int) -> None:
        self.progress.setValue(100 if total == 0 else round(done * 100 / total))
        self.log.setPlainText(f"Проверено сценариев: {done}/{total}")

    def calculation_finished(self, result: object, output_path: str) -> None:
        if isinstance(result, pd.DataFrame):
            self.show_preview(result)
        self.progress.setValue(100)
        self.log.setPlainText(f"Готово. Результат сохранён:\n{output_path}")
        self.run_button.setEnabled(True)

    def calculation_failed(self, message: str) -> None:
        self.log.setPlainText(f"Ошибка: {message}")
        self.run_button.setEnabled(True)
        show_dialog(self, QMessageBox.Icon.Critical, "Ошибка расчёта", message)

    def thread_finished(self) -> None:
        self.worker = None
        self.worker_thread = None


class SingleTab(QWidget):
    MACHINE_FIELDS = (
        ("l", "Захват, м"),
        ("U", "Раб. скорость, м/с"),
        ("Vtravel", "Переезд, м/с"),
        ("Vturn", "Разворот, м/с"),
        ("hx", "Длина, м"),
        ("hy", "Ширина, м"),
        ("R", "Радиус поворота, м"),
        ("coord_x", "Старт X, м"),
        ("coord_y", "Старт Y, м"),
        ("orient", "Ориентация, °"),
    )

    def __init__(self) -> None:
        super().__init__()
        self.worker_thread: QThread | None = None
        self.worker: SingleWorker | None = None
        self.current_row: pd.Series | None = None
        self.current_evaluation: ScenarioEvaluation | None = None
        self.inputs: dict[str, QDoubleSpinBox] = {}

        self.axis = QComboBox()
        self.axis.addItem("Вдоль X", "x")
        self.axis.addItem("Вдоль Y", "y")
        self.axis.setMinimumWidth(96)
        self.axis.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.field_x = make_spin(FIELD_X)
        self.field_y = make_spin(FIELD_Y)
        self.turn_zone = make_spin(TURN_ZONE_WIDTH)
        self.allowed_gap = make_spin(
            0.0,
            minimum=0.0,
            maximum=100.0,
            decimals=2,
        )
        self.finish_mode = QComboBox()
        self.finish_mode.addItem("Остановка у границы", False)
        self.finish_mode.addItem("Возврат к старту", True)
        self.finish_mode.setMinimumWidth(96)
        self.finish_mode.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        field_header = QLabel("Поле")
        field_header.setObjectName("groupHeader")
        field_header.setProperty("accent", "field")
        field_header.setFixedHeight(36)
        field_grid = QGridLayout()
        field_grid.setContentsMargins(0, 0, 0, 0)
        field_grid.setHorizontalSpacing(10)
        field_grid.setVerticalSpacing(4)
        field_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        field_values = (
            ("Направление проходов", self.axis),
            ("Размер поля X, м", self.field_x),
            ("Размер поля Y, м", self.field_y),
            ("Ширина разворотной зоны, м", self.turn_zone),
            ("Допустимые пропуски, %", self.allowed_gap),
            ("Завершение маршрута", self.finish_mode),
        )
        for item_index, (label_text, editor) in enumerate(field_values):
            column = item_index % 2
            label_row = (item_index // 2) * 2
            label = QLabel(label_text)
            label.setObjectName("compactLabel")
            field_grid.addWidget(label, label_row, column)
            field_grid.addWidget(editor, label_row + 1, column)
            field_grid.setColumnStretch(column, 1)
        field_layout = QVBoxLayout()
        field_layout.setSpacing(8)
        field_layout.addWidget(field_header)
        field_layout.addLayout(field_grid)
        field_group = QGroupBox()
        field_group.setObjectName("fieldGroup")
        field_group.setLayout(field_layout)
        field_group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )

        defaults = {
            1: (6.0, 2.0, 2.5, 1.5, 5.0, 3.0, 2.5, 0.0, 3.0, 0.0),
            2: (6.0, 2.0, 2.5, 1.5, 5.0, 3.0, 2.5, 0.0, 117.0, 0.0),
        }
        machines_layout = QHBoxLayout()
        machines_layout.setSpacing(10)
        machines_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for machine_index in (1, 2):
            machine_grid = QGridLayout()
            machine_grid.setContentsMargins(0, 0, 0, 0)
            machine_grid.setHorizontalSpacing(10)
            machine_grid.setVerticalSpacing(4)
            machine_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
            machine_header = QLabel(f"Машина {machine_index}")
            machine_header.setObjectName("groupHeader")
            machine_header.setProperty("accent", f"machine{machine_index}")
            machine_header.setFixedHeight(36)
            for item_index, ((field_name, label), value) in enumerate(zip(
                self.MACHINE_FIELDS, defaults[machine_index], strict=True
            )):
                minimum = -360.0 if field_name == "orient" else 0.0
                spin = make_spin(value, minimum=minimum)
                spin.setFixedHeight(28)
                if field_name == "orient":
                    spin.setMaximum(360.0)
                self.inputs[f"{field_name}{machine_index}"] = spin
                column = item_index % 2
                label_row = (item_index // 2) * 2
                input_label = QLabel(label)
                input_label.setObjectName("compactLabel")
                machine_grid.addWidget(input_label, label_row, column)
                machine_grid.addWidget(spin, label_row + 1, column)
                machine_grid.setColumnStretch(column, 1)
            machine_layout = QVBoxLayout()
            machine_layout.setSpacing(8)
            machine_layout.addWidget(machine_header)
            machine_layout.addLayout(machine_grid)
            group = QGroupBox()
            group.setObjectName(f"machine{machine_index}")
            group.setLayout(machine_layout)
            group.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Maximum,
            )
            machines_layout.addWidget(group)

        self.winner_label = QLabel("Лучший план ещё не рассчитан")
        self.winner_label.setWordWrap(True)
        self.winner_label.setObjectName("winnerCard")
        self.winner_label.setMinimumHeight(68)
        self.results = QTableWidget(3, 7)
        self.results.setObjectName("resultTable")
        self.results.setMinimumHeight(175)
        self.results.setHorizontalHeaderLabels(
            [
                "Стратег.",
                "Статус",
                "Итог, с",
                "М1/М2, с",
                "Раб./пер.",
                "Повор./зад.",
                "Перекр./пр.",
            ]
        )
        self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results.setAlternatingRowColors(True)
        self.results.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results.setWordWrap(False)
        self.results.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        results_header = self.results.horizontalHeader()
        if results_header is not None:
            results_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_vertical_header = self.results.verticalHeader()
        if results_vertical_header is not None:
            results_vertical_header.setVisible(False)
            results_vertical_header.setDefaultSectionSize(38)
        self.canvas = RouteCanvas()
        self.calculate_button = QPushButton("Рассчитать частный случай")
        self.save_button = QPushButton("Сохранить XLSX-отчёт")
        self.save_button.setProperty("variant", "secondary")
        self.save_button.setEnabled(False)
        self.save_image_button = QPushButton("Сохранить схему PNG")
        self.save_image_button.setProperty("variant", "secondary")
        self.save_image_button.setEnabled(False)
        self.calculate_button.clicked.connect(self.start_calculation)
        self.save_button.clicked.connect(self.save_report)
        self.save_image_button.clicked.connect(self.save_image)
        buttons = QHBoxLayout()
        buttons.addWidget(self.calculate_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.save_image_button)
        buttons.addStretch()

        input_card, input_card_layout = make_card()
        input_card.setMinimumWidth(620)
        input_card_layout.addWidget(make_section_title("Числовые параметры"))
        input_hint = QLabel(
            "Выбранное направление применяется ко всем стратегиям. Чтобы сравнить "
            "варианты, выполните расчёт отдельно для X и Y."
        )
        input_hint.setObjectName("mutedText")
        input_hint.setWordWrap(True)
        input_card_layout.addWidget(input_hint)
        input_card_layout.addWidget(field_group)
        input_card_layout.addLayout(machines_layout)
        input_card_layout.addLayout(buttons)
        input_card_layout.addStretch(1)

        result_card, result_layout = make_card()
        result_card.setMinimumWidth(500)
        result_layout.addWidget(make_section_title("Результат и выбранный маршрут"))
        result_layout.addWidget(self.winner_label)
        result_layout.addWidget(self.results)
        result_layout.addWidget(self.canvas, 1)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(input_card)
        splitter.addWidget(result_card)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 720])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(
            make_page_header(
                "Расчёт частного сценария",
                "Задайте фактическую геометрию поля и характеристики двух машин. "
                "Приложение сравнит стратегии, объяснит ограничения и нарисует маршрут.",
            )
        )
        layout.addWidget(splitter)

    def build_row(self) -> pd.Series:
        values: dict[str, object] = {
            "scenario_id": "частный случай",
            "s": self.axis.currentData(),
            "field_x": self.field_x.value(),
            "field_y": self.field_y.value(),
            "turn_zone_width": self.turn_zone.value(),
            "allowed_gap_percent": self.allowed_gap.value(),
            "return_to_start": bool(self.finish_mode.currentData()),
        }
        for machine_index in (1, 2):
            for field_name, _ in self.MACHINE_FIELDS:
                column = (
                    f"coord{machine_index}_{field_name.removeprefix('coord_')}"
                    if field_name.startswith("coord_")
                    else f"{field_name}{machine_index}"
                )
                values[column] = self.inputs[
                    f"{field_name}{machine_index}"
                ].value()
        return pd.Series(values, dtype=object)

    def start_calculation(self) -> None:
        self.current_row = self.build_row()
        self.current_evaluation = None
        self.calculate_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.save_image_button.setEnabled(False)
        self.winner_label.setText("Выполняется расчёт…")
        thread = QThread(self)
        worker = SingleWorker(self.current_row)
        self.worker_thread = thread
        self.worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self.calculation_finished)
        worker.failed.connect(self.calculation_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self.thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def calculation_finished(self, value: object) -> None:
        if not isinstance(value, ScenarioEvaluation):
            self.calculation_failed("Получен неизвестный формат результата")
            return
        self.current_evaluation = value
        result = value.result

        def pair_text(first: object, second: object) -> str:
            left = "—" if first is None else str(first)
            right = "—" if second is None else str(second)
            return f"{left} / {right}"

        for row_index, (plan_name, suffix, label) in enumerate(STRATEGIES):
            plan = value.plans[plan_name]
            short_label = "По производит." if label == "По производительности" else label
            table_label = f"{short_label} ({plan.axis.upper()})"
            cells = (
                table_label,
                plan.status,
                result[f"T_{suffix}"],
                pair_text(result[f"T1_{suffix}"], result[f"T2_{suffix}"]),
                pair_text(result[f"work_{suffix}"], result[f"deadhead_{suffix}"]),
                pair_text(result[f"turns_{suffix}"], result[f"delay_{suffix}"]),
                pair_text(result[f"overlap_{suffix}"], result[f"uncovered_{suffix}"]),
            )
            for column_index, cell in enumerate(cells):
                item = QTableWidgetItem("—" if cell is None else str(cell))
                item.setToolTip("—" if cell is None else str(cell))
                if column_index >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results.setItem(row_index, column_index, item)
        winner = value.winner
        if winner is None:
            self.winner_label.setText(
                "Допустимый план не найден. Причины приведены в таблице."
            )
        else:
            self.winner_label.setText(
                f"Лучший план: {strategy_display_name(winner.name)}; "
                f"направление {winner.axis.upper()}; "
                f"время завершения: {winner.makespan:.1f} с; "
                f"покрытие: {winner.coverage_percent:.2f}%"
            )
        self.canvas.set_route(winner, value.geometry)
        self.calculate_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.save_image_button.setEnabled(winner is not None)

    def calculation_failed(self, message: str) -> None:
        self.winner_label.setText(f"Расчёт невозможен: {message}")
        self.canvas.set_route(None, None)
        self.calculate_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.save_image_button.setEnabled(False)
        show_dialog(self, QMessageBox.Icon.Critical, "Ошибка расчёта", message)

    def thread_finished(self) -> None:
        self.worker = None
        self.worker_thread = None

    def save_report(self) -> None:
        if self.current_row is None or self.current_evaluation is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить расчёт", "private_case_result.xlsx", "Excel (*.xlsx)"
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".xlsx":
            path = path.with_suffix(".xlsx")
        try:
            save_private_report(self.current_row, self.current_evaluation, path)
            show_dialog(
                self,
                QMessageBox.Icon.Information,
                "Отчёт сохранён",
                f"Файл сохранён:\n{path}",
            )
        except Exception as error:  # noqa: BLE001
            show_dialog(
                self, QMessageBox.Icon.Critical, "Ошибка сохранения", str(error)
            )

    def save_image(self) -> None:
        if self.current_evaluation is None or self.current_evaluation.winner is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить схему маршрутов",
            "route_map.png",
            "PNG (*.png)",
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        if self.canvas.save_png(path):
            show_dialog(
                self,
                QMessageBox.Icon.Information,
                "Схема сохранена",
                f"Изображение сохранено:\n{path}",
            )
        else:
            show_dialog(
                self,
                QMessageBox.Icon.Critical,
                "Ошибка сохранения",
                "Не удалось сохранить PNG",
            )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Inagrika — планировщик уборки")
        self.setMinimumSize(1280, 900)
        self.resize(1440, 960)
        tabs = QTabWidget()
        tabs.setObjectName("mainTabs")
        tabs.addTab(TableTab(), "Загрузить таблицу")
        tabs.addTab(SingleTab(), "Рассчитать частный случай")
        self.setCentralWidget(tabs)
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(
                "Входные данные числовые; знаки сравнения вычисляются автоматически"
            )


def configure_application(app: QApplication) -> None:
    """Настроить шрифт, палитру и стили приложения."""

    app.setStyle("Fusion")
    app.setFont(QFont(load_ui_font(), 10))
    app.setStyleSheet(
        """
        QMainWindow, QTabWidget#mainTabs > QWidget { background: #eef2f7; }
        QWidget { color: #0f172a; }
        QTabWidget::pane { border: none; background: #eef2f7; }
        QTabBar::tab {
            background: #dbe4f0; color: #475569; padding: 12px 24px;
            margin-right: 3px; border-top-left-radius: 8px;
            border-top-right-radius: 8px; font-weight: 600;
        }
        QTabBar::tab:selected { background: #ffffff; color: #1d4ed8; }
        QTabBar::tab:hover:!selected { background: #e8eef6; color: #1e3a8a; }
        QFrame#pageHeader {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #1d4ed8, stop:1 #0f766e);
            border-radius: 12px;
        }
        QLabel#pageTitle { color: white; font-size: 24px; font-weight: 700; }
        QLabel#pageSubtitle { color: #dbeafe; font-size: 11px; }
        QFrame#card {
            background: #ffffff; border: 1px solid #dbe3ed; border-radius: 10px;
        }
        QLabel#sectionTitle { font-size: 14px; font-weight: 700; color: #172554; }
        QLabel#mutedText { color: #64748b; }
        QLabel#compactLabel { color: #334155; font-size: 9pt; font-weight: 500; }
        QLabel#groupHeader {
            color: white; border-radius: 6px; padding: 8px 12px;
            font-size: 13px; font-weight: 700;
        }
        QLabel#groupHeader[accent="field"] { background: #0f766e; }
        QLabel#groupHeader[accent="machine1"] { background: #be123c; }
        QLabel#groupHeader[accent="machine2"] { background: #1d4ed8; }
        QLabel#badge {
            background: #e0f2fe; color: #075985; border: 1px solid #bae6fd;
            border-radius: 12px; padding: 5px 11px; font-weight: 600;
        }
        QLabel#winnerCard {
            background: #ecfdf5; color: #14532d; border: 1px solid #a7f3d0;
            border-radius: 8px; padding: 11px 14px; font-size: 14px;
            font-weight: 700;
        }
        QGroupBox {
            font-weight: 700; border: 1px solid #cbd5e1; border-radius: 8px;
            margin-top: 0; padding: 9px; background: #f8fafc;
        }
        QGroupBox#machine1 { border-top: 3px solid #e11d48; }
        QGroupBox#machine2 { border-top: 3px solid #2563eb; }
        QPushButton {
            min-height: 36px; padding: 0 16px; border: none; border-radius: 7px;
            background: #2563eb; color: white; font-weight: 700;
        }
        QPushButton:hover { background: #1d4ed8; }
        QPushButton:pressed { background: #1e40af; }
        QPushButton:disabled { background: #cbd5e1; color: #64748b; }
        QPushButton[variant="secondary"] {
            background: #ffffff; color: #1e3a8a; border: 1px solid #93c5fd;
        }
        QPushButton[variant="secondary"]:hover { background: #eff6ff; }
        QLineEdit, QComboBox, QTextEdit {
            min-height: 34px; background: #ffffff; border: 1px solid #cbd5e1;
            border-radius: 6px; padding: 2px 8px; color: #0f172a;
            selection-background-color: #bfdbfe; selection-color: #0f172a;
        }
        QComboBox QAbstractItemView {
            background: #ffffff; color: #0f172a; border: 1px solid #94a3b8;
            outline: 0; selection-background-color: #dbeafe;
            selection-color: #0f172a;
        }
        QComboBox QAbstractItemView::item {
            min-height: 30px; padding: 4px 8px;
        }
        QDoubleSpinBox {
            background: #ffffff; border: 1px solid #cbd5e1;
            border-radius: 6px; padding: 1px 8px;
            selection-background-color: #bfdbfe;
        }
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            width: 0; height: 0; border: none; background: transparent;
        }
        QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {
            border: 2px solid #60a5fa;
        }
        QLineEdit:read-only { background: #f8fafc; color: #475569; }
        QTableWidget {
            background: #ffffff; alternate-background-color: #f8fafc;
            border: 1px solid #dbe3ed; border-radius: 7px;
            gridline-color: #e2e8f0; selection-background-color: #dbeafe;
            selection-color: #0f172a;
        }
        QHeaderView::section {
            background: #eaf0f7; color: #334155; border: none;
            border-right: 1px solid #d7e0ea; border-bottom: 1px solid #cbd5e1;
            padding: 8px; font-weight: 700;
        }
        QProgressBar {
            min-height: 20px; border: 1px solid #cbd5e1; border-radius: 7px;
            background: #f1f5f9; text-align: center; color: #334155;
        }
        QProgressBar::chunk {
            border-radius: 6px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #2563eb, stop:1 #14b8a6);
        }
        QWidget#routeCanvas {
            border: 1px solid #cbd5e1; border-radius: 9px; background: #ffffff;
        }
        QSplitter::handle { background: #dbe3ed; height: 5px; }
        QStatusBar { background: #ffffff; color: #475569; border-top: 1px solid #dbe3ed; }
        QScrollBar:vertical {
            background: #f1f5f9; width: 12px; margin: 0; border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #94a3b8; min-height: 28px; border-radius: 6px;
        }
        """
    )


def main() -> None:
    app = QApplication(sys.argv)
    configure_application(app)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()


