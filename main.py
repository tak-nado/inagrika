from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# Геометрия поля и разворотных зон, м.
FIELD_X = 240.0
FIELD_Y = 120.0
TURN_ZONE_WIDTH = 10.0

# Базовые характеристики машин.
BASE_HEADER = 6.0
BASE_WORK_SPEED = 2.0
BASE_LENGTH = 5.0
BASE_WIDTH = 3.0

HEADER_VARIANTS = {"=": BASE_HEADER, ">": 8.0, "<": 4.5}
SPEED_VARIANTS = {"=": BASE_WORK_SPEED, ">": 3.0, "<": 1.5}
LENGTH_VARIANTS = {"=": BASE_LENGTH, ">": 6.0, "<": 4.0}
WIDTH_VARIANTS = {"=": BASE_WIDTH, ">": 3.5, "<": 2.5}

# Явные допущения расчётной модели.
TRAVEL_SPEED_FACTOR = 1.25
TURN_SPEED_FACTOR = 0.75
MINIMUM_TURN_RADIUS = 2.5
SAFETY_MARGIN = 0.25
TIME_TIE_FRACTION = 0.001
START_DELAY_STEP = 5.0
MAX_START_DELAY = 180.0
ARC_STEP_DEGREES = 10.0
EPS = 1e-9

Point = tuple[float, float]


@dataclass(frozen=True)
class Machine:
    """Физические и скоростные характеристики одного комбайна.

    Координаты относятся к центру машины. Ориентация хранится в радианах:
    0 соответствует движению вдоль положительного направления оси OX.
    """

    header: float
    work_speed: float
    travel_speed: float
    turn_speed: float
    length: float
    width: float
    turn_radius: float
    start: Point
    orientation: float

    @property
    def collision_radius(self) -> float:
        """Вернуть радиус окружности, полностью покрывающей машину и жатку."""

        collision_width = max(self.width, self.header)
        return math.hypot(self.length / 2.0, collision_width / 2.0) + SAFETY_MARGIN


@dataclass(frozen=True)
class FieldGeometry:
    """Геометрия поля для выбранного направления рабочих проходов.

    ``axis='x'`` означает движение вдоль длины поля, ``axis='y'`` — вдоль
    ширины. С обоих концов рабочего прохода исключаются разворотные зоны.
    """

    axis: str

    @property
    def run_length(self) -> float:
        """Вернуть полезную длину одного рабочего прохода без разворотных зон."""

        full = FIELD_X if self.axis == "x" else FIELD_Y
        return full - 2.0 * TURN_ZONE_WIDTH

    @property
    def sweep_width(self) -> float:
        """Вернуть ширину поля, которую необходимо покрыть проходами."""

        return FIELD_Y if self.axis == "x" else FIELD_X

    def entry_point(self, lateral: float, far_end: bool) -> Point:
        """Получить точку входа в проход на ближнем или дальнем краю посева."""

        if self.axis == "x":
            return (FIELD_X - TURN_ZONE_WIDTH if far_end else TURN_ZONE_WIDTH, lateral)
        return (lateral, FIELD_Y - TURN_ZONE_WIDTH if far_end else TURN_ZONE_WIDTH)

    def outer_point(self, lateral: float, far_end: bool) -> Point:
        """Получить точку на внешней границе соответствующей разворотной зоны."""

        if self.axis == "x":
            return (FIELD_X if far_end else 0.0, lateral)
        return (lateral, FIELD_Y if far_end else 0.0)


@dataclass(frozen=True)
class Segment:
    """Один прямолинейный или неподвижный временной участок маршрута."""

    begin: Point
    end: Point
    begin_time: float
    end_time: float
    kind: str


@dataclass
class Route:
    """Полный маршрут одной машины и рассчитанные характеристики движения."""

    segments: list[Segment]
    duration: float
    start: Point
    finish: Point
    work_distance: float
    deadhead_distance: float
    turn_distance: float
    pass_count: int
    turn_count: int
    reverse_order: bool
    start_at_far_end: bool


@dataclass
class Plan:
    """Результат проверки одной стратегии для пары комбайнов.

    Недопустимый план содержит описание причины в ``status``. Для допустимого
    плана ``makespan`` равен времени завершения последней машины.
    """

    name: str
    valid: bool = False
    status: str = "нет допустимого маршрута"
    makespan: float = math.inf
    time_1: float = math.inf
    time_2: float = math.inf
    delay_1: float = 0.0
    delay_2: float = 0.0
    deadhead: float = math.inf
    overlap_area: float = math.inf
    turns: int = 0
    route_1: Route | None = None
    route_2: Route | None = None


def distance(first: Point, second: Point) -> float:
    """Вычислить евклидово расстояние между двумя точками."""

    return math.hypot(first[0] - second[0], first[1] - second[1])


def heading(first: Point, second: Point) -> float:
    """Вычислить угол направления движения из первой точки во вторую."""

    return math.atan2(second[1] - first[1], second[0] - first[0])


def angle_distance(first: float, second: float) -> float:
    """Вернуть минимальный модуль поворота между двумя углами в радианах."""

    difference = (second - first + math.pi) % (2.0 * math.pi) - math.pi
    return abs(difference)


class RouteBuilder:
    """Последовательно строит маршрут и накапливает время и длины путей."""

    def __init__(self, machine: Machine) -> None:
        """Начать маршрут в стартовой точке с заданной ориентацией машины."""

        self.machine = machine
        self.position = machine.start
        self.heading = machine.orientation
        self.time = 0.0
        self.segments: list[Segment] = []
        self.work_distance = 0.0
        self.deadhead_distance = 0.0
        self.turn_distance = 0.0

    def rotate_to(self, target_heading: float, kind: str) -> None:
        """Добавить поворот на месте к требуемому направлению.

        Время поворота определяется длиной соответствующей дуги минимального
        радиуса и разворотной скоростью машины.
        """

        angle = angle_distance(self.heading, target_heading)
        if angle <= EPS:
            self.heading = target_heading
            return
        duration = angle * self.machine.turn_radius / self.machine.turn_speed
        self.segments.append(
            Segment(self.position, self.position, self.time, self.time + duration, kind)
        )
        self.time += duration
        self.heading = target_heading

    def move_to(self, target: Point, speed: float, kind: str,
                rotate_before: bool = True,
                path_length: float | None = None) -> None:
        """Добавить прямолинейное движение к точке с указанной скоростью.

        По умолчанию перед движением добавляется необходимый поворот на месте.
        Для хорд радиальной дуги ``rotate_before`` отключается, а ``path_length``
        хранит фактическую длину дуги вместо длины хорды. ``kind`` определяет,
        к какой категории будет отнесён пройденный путь.
        """

        chord_length = distance(self.position, target)
        if chord_length <= EPS:
            self.position = target
            return
        target_heading = heading(self.position, target)
        if rotate_before:
            self.rotate_to(target_heading, kind)
        length = chord_length if path_length is None else path_length
        duration = length / speed
        self.segments.append(
            Segment(self.position, target, self.time, self.time + duration, kind)
        )
        self.time += duration
        self.position = target
        self.heading = target_heading
        if kind == "work":
            self.work_distance += length
        elif kind == "turn":
            self.turn_distance += length
        else:
            self.deadhead_distance += length

    def follow(self, points: list[Point], speed: float, kind: str,
               rotate_before: bool = True) -> None:
        """Последовательно пройти через список контрольных точек."""

        for point in points:
            self.move_to(point, speed, kind, rotate_before=rotate_before)


def headland_is_feasible(machine: Machine, outward_extent: float) -> bool:
    """Проверить, помещаются ли траектория и корпус в разворотной зоне."""

    required_width = outward_extent + machine.collision_radius
    return TURN_ZONE_WIDTH + EPS >= required_width


def local_to_global(origin: Point, outward: Point, transverse: Point,
                    local: Point, side: float = 1.0) -> Point:
    """Перевести точку локального разворота в координаты поля.

    Локальная ось X направлена наружу от рабочего участка, ось Y — вдоль
    поперечного направления проходов. ``side`` зеркально отражает манёвр для
    перехода к полосе с меньшей поперечной координатой.
    """

    local_x, local_y = local
    return (
        origin[0] + local_x * outward[0] + side * local_y * transverse[0],
        origin[1] + local_x * outward[1] + side * local_y * transverse[1],
    )


def append_local_arc(builder: RouteBuilder, origin: Point, outward: Point,
                     transverse: Point, side: float, center: Point,
                     radius: float, start_angle: float,
                     sweep_angle: float) -> None:
    """Добавить радиальную дугу как набор коротких хорд без поворотов на месте."""

    step = math.radians(ARC_STEP_DEGREES)
    part_count = max(1, math.ceil(abs(sweep_angle) / step))
    previous_angle = start_angle
    for index in range(1, part_count + 1):
        angle = start_angle + sweep_angle * index / part_count
        local_point = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
        target = local_to_global(origin, outward, transverse, local_point, side)
        arc_length = radius * abs(angle - previous_angle)
        builder.move_to(
            target,
            builder.machine.turn_speed,
            "turn",
            rotate_before=False,
            path_length=arc_length,
        )
        previous_angle = angle


def append_radial_turn(builder: RouteBuilder, geometry: FieldGeometry,
                       current_lateral: float, next_lateral: float,
                       far_end: bool) -> bool:
    """Соединить соседние проходы непрерывным радиальным разворотом.

    При расстоянии не меньше двух минимальных радиусов строятся две четверти
    окружности и прямой поперечный участок. Для более близких проходов строится
    петлевой RLR-манёвр из трёх дуг того же минимального радиуса.

    Returns:
        ``False``, если траектория с учётом половины длины машины не помещается
        в разворотной зоне.
    """

    shift = next_lateral - current_lateral
    shift_size = abs(shift)
    if shift_size <= EPS:
        return False

    side = 1.0 if shift > 0.0 else -1.0
    if geometry.axis == "x":
        outward = (1.0 if far_end else -1.0, 0.0)
        transverse = (0.0, 1.0)
    else:
        outward = (0.0, 1.0 if far_end else -1.0)
        transverse = (1.0, 0.0)

    radius = builder.machine.turn_radius
    origin = builder.position

    if shift_size + EPS >= 2.0 * radius:
        outward_extent = radius
        if not headland_is_feasible(builder.machine, outward_extent):
            return False

        append_local_arc(
            builder, origin, outward, transverse, side,
            (0.0, radius), radius, -math.pi / 2.0, math.pi / 2.0,
        )
        straight_end = (radius, shift_size - radius)
        builder.move_to(
            local_to_global(origin, outward, transverse, straight_end, side),
            builder.machine.turn_speed,
            "turn",
            rotate_before=False,
        )
        append_local_arc(
            builder, origin, outward, transverse, side,
            (0.0, shift_size - radius), radius, 0.0, math.pi / 2.0,
        )
    else:
        center_distance = shift_size + 2.0 * radius
        half_distance = center_distance / 2.0
        center_offset = math.sqrt(max(0.0, (2.0 * radius) ** 2 - half_distance ** 2))
        outward_extent = center_offset + radius
        if not headland_is_feasible(builder.machine, outward_extent):
            return False

        tangent_angle = math.atan2(center_distance, 2.0 * center_offset)
        append_local_arc(
            builder, origin, outward, transverse, side,
            (0.0, -radius), radius, math.pi / 2.0,
            tangent_angle - math.pi / 2.0,
        )
        append_local_arc(
            builder, origin, outward, transverse, side,
            (center_offset, shift_size / 2.0), radius,
            -math.pi + tangent_angle,
            2.0 * math.pi - 2.0 * tangent_angle,
        )
        append_local_arc(
            builder, origin, outward, transverse, side,
            (0.0, shift_size + radius), radius, -tangent_angle,
            tangent_angle - math.pi / 2.0,
        )

    expected_finish = geometry.entry_point(next_lateral, far_end)
    builder.position = expected_finish
    builder.heading = math.atan2(-outward[1], -outward[0])
    return True


def perimeter_path(start: Point, entry: Point, geometry: FieldGeometry,
                   far_end: bool) -> list[Point]:
    """Построить порожний путь к проходу вдоль границы поля.

    Если требуемый проход начинается на противоположной стороне, машина идёт
    через ближайший угол поля. Это исключает прямое пересечение неубранного
    посева по диагонали.
    """

    outer_entry = geometry.outer_point(
        entry[1] if geometry.axis == "x" else entry[0], far_end
    )
    if geometry.axis == "x":
        current_far = start[0] > FIELD_X / 2.0
        if current_far == far_end:
            return [outer_entry, entry]
        edge = min((0.0, FIELD_Y), key=lambda y: abs(start[1] - y) + abs(entry[1] - y))
        opposite_x = FIELD_X if far_end else 0.0
        return [(start[0], edge), (opposite_x, edge), outer_entry, entry]

    current_far = start[1] > FIELD_Y / 2.0
    if current_far == far_end:
        return [outer_entry, entry]
    edge = min((0.0, FIELD_X), key=lambda x: abs(start[0] - x) + abs(entry[0] - x))
    opposite_y = FIELD_Y if far_end else 0.0
    return [(edge, start[1]), (edge, opposite_y), outer_entry, entry]


def build_route(machine: Machine, geometry: FieldGeometry, passes: list[float],
                reverse_order: bool, start_at_far_end: bool) -> Route | None:
    """Построить полный змееобразный маршрут одной машины.

    Args:
        machine: характеристики и стартовое состояние машины.
        geometry: направление проходов и размеры рабочей области.
        passes: поперечные координаты назначенных машине проходов.
        reverse_order: начинать с прохода с большей поперечной координатой.
        start_at_far_end: начинать рабочее движение с дальнего края поля.

    Returns:
        Готовый маршрут либо ``None``, если разворотная зона слишком узкая.
    """

    if not passes:
        return Route([], 0.0, machine.start, machine.start, 0.0, 0.0, 0.0,
                     0, 0, reverse_order, start_at_far_end)

    ordered = list(reversed(passes)) if reverse_order else list(passes)
    builder = RouteBuilder(machine)
    boundary = start_at_far_end
    first_entry = geometry.entry_point(ordered[0], boundary)
    builder.follow(
        perimeter_path(machine.start, first_entry, geometry, boundary),
        machine.travel_speed,
        "deadhead",
    )

    for index, lateral in enumerate(ordered):
        work_end = geometry.entry_point(lateral, not boundary)
        builder.move_to(work_end, machine.work_speed, "work")
        boundary = not boundary

        if index + 1 < len(ordered):
            if not append_radial_turn(
                builder, geometry, lateral, ordered[index + 1], boundary
            ):
                return None

    # Освобождаем рабочую и разворотную зоны после последнего прохода.
    final_parking = geometry.outer_point(ordered[-1], boundary)
    builder.move_to(final_parking, machine.travel_speed, "deadhead")

    return Route(
        segments=builder.segments,
        duration=builder.time,
        start=machine.start,
        finish=builder.position,
        work_distance=builder.work_distance,
        deadhead_distance=builder.deadhead_distance,
        turn_distance=builder.turn_distance,
        pass_count=len(ordered),
        turn_count=max(0, len(ordered) - 1),
        reverse_order=reverse_order,
        start_at_far_end=start_at_far_end,
    )


def route_options(machine: Machine, geometry: FieldGeometry,
                  passes: list[float]) -> list[Route]:
    """Сформировать четыре варианта маршрута по порядку полос и стороне входа."""

    options: list[Route] = []
    for reverse_order in (False, True):
        for start_at_far_end in (False, True):
            route = build_route(machine, geometry, passes, reverse_order, start_at_far_end)
            if route is not None:
                options.append(route)
    return options


def shifted_timeline(route: Route, delay: float, horizon: float) -> list[Segment]:
    """Создать временную шкалу маршрута с задержкой старта и парковкой.

    После завершения маршрута машина остаётся неподвижной в точке парковки до
    общего времени завершения пары.
    """

    timeline: list[Segment] = []
    if delay > EPS:
        timeline.append(Segment(route.start, route.start, 0.0, delay, "wait"))
    for segment in route.segments:
        timeline.append(
            Segment(
                segment.begin,
                segment.end,
                segment.begin_time + delay,
                segment.end_time + delay,
                segment.kind,
            )
        )
    finish_time = delay + route.duration
    if finish_time < horizon - EPS:
        timeline.append(Segment(route.finish, route.finish, finish_time, horizon, "park"))
    if not timeline and horizon > EPS:
        timeline.append(Segment(route.start, route.start, 0.0, horizon, "park"))
    return timeline


def position_on_segment(segment: Segment, moment: float) -> Point:
    """Вычислить положение машины на сегменте в заданный момент времени."""

    duration = segment.end_time - segment.begin_time
    if duration <= EPS:
        return segment.end
    part = (moment - segment.begin_time) / duration
    part = min(1.0, max(0.0, part))
    return (
        segment.begin[0] + (segment.end[0] - segment.begin[0]) * part,
        segment.begin[1] + (segment.end[1] - segment.begin[1]) * part,
    )


def segment_velocity(segment: Segment) -> Point:
    """Вернуть постоянный вектор скорости на линейном сегменте."""

    duration = segment.end_time - segment.begin_time
    if duration <= EPS:
        return (0.0, 0.0)
    return (
        (segment.end[0] - segment.begin[0]) / duration,
        (segment.end[1] - segment.begin[1]) / duration,
    )


def segments_collide(first: Segment, second: Segment, radius_sum: float) -> bool:
    """Проверить столкновение на пересекающемся временном интервале.

    Минимальное расстояние между двумя равномерно движущимися точками находится
    аналитически. Столкновение фиксируется, если оно не превышает сумму
    безопасных радиусов машин.
    """

    begin = max(first.begin_time, second.begin_time)
    end = min(first.end_time, second.end_time)
    if end < begin - EPS:
        return False

    first_position = position_on_segment(first, begin)
    second_position = position_on_segment(second, begin)
    first_velocity = segment_velocity(first)
    second_velocity = segment_velocity(second)
    relative_position = (
        first_position[0] - second_position[0],
        first_position[1] - second_position[1],
    )
    relative_velocity = (
        first_velocity[0] - second_velocity[0],
        first_velocity[1] - second_velocity[1],
    )
    duration = max(0.0, end - begin)
    velocity_squared = (
        relative_velocity[0] ** 2 + relative_velocity[1] ** 2
    )
    if velocity_squared <= EPS:
        closest_time = 0.0
    else:
        closest_time = -(
            relative_position[0] * relative_velocity[0]
            + relative_position[1] * relative_velocity[1]
        ) / velocity_squared
        closest_time = min(duration, max(0.0, closest_time))

    dx = relative_position[0] + relative_velocity[0] * closest_time
    dy = relative_position[1] + relative_velocity[1] * closest_time
    return dx * dx + dy * dy <= radius_sum * radius_sum


def routes_collide(machine_1: Machine, route_1: Route, delay_1: float,
                   machine_2: Machine, route_2: Route, delay_2: float) -> bool:
    """Проверить две полные траектории с учётом задержек и парковки."""

    radius_sum = machine_1.collision_radius + machine_2.collision_radius
    if distance(route_1.start, route_2.start) <= radius_sum:
        return True

    horizon = max(delay_1 + route_1.duration, delay_2 + route_2.duration)
    first = shifted_timeline(route_1, delay_1, horizon)
    second = shifted_timeline(route_2, delay_2, horizon)
    index_1 = 0
    index_2 = 0

    while index_1 < len(first) and index_2 < len(second):
        segment_1 = first[index_1]
        segment_2 = second[index_2]
        if segments_collide(segment_1, segment_2, radius_sum):
            return True
        if segment_1.end_time < segment_2.end_time - EPS:
            index_1 += 1
        elif segment_2.end_time < segment_1.end_time - EPS:
            index_2 += 1
        else:
            index_1 += 1
            index_2 += 1
    return False


def find_safe_delays(machine_1: Machine, route_1: Route,
                     machine_2: Machine, route_2: Route) -> tuple[float, float] | None:
    """Найти лучшую безопасную задержку запуска одной из машин.

    Проверяются задержки с фиксированным шагом от нуля до
    ``MAX_START_DELAY``. Возвращается пара задержек с минимальным временем
    завершения либо ``None``, если все варианты приводят к столкновению.
    """

    delays = [0.0]
    delay = START_DELAY_STEP
    while delay <= MAX_START_DELAY + EPS:
        delays.append(delay)
        delay += START_DELAY_STEP

    best: tuple[float, float] | None = None
    best_makespan = math.inf
    for current in delays:
        variants = [(0.0, 0.0)] if current == 0.0 else [(0.0, current), (current, 0.0)]
        for pair in variants:
            makespan = max(pair[0] + route_1.duration, pair[1] + route_2.duration)
            if makespan >= best_makespan - EPS:
                continue
            if not routes_collide(machine_1, route_1, pair[0], machine_2, route_2, pair[1]):
                best = pair
                best_makespan = makespan
    return best


def plan_is_better(candidate: Plan, current: Plan | None) -> bool:
    """Сравнить планы по критериям кейса.

    Сначала минимизируется время завершения последней машины. При близком
    времени предпочтение получает меньший порожний путь, затем перекрытие.
    """

    if not candidate.valid:
        return False
    if current is None or not current.valid:
        return True
    tolerance = max(candidate.makespan, current.makespan) * TIME_TIE_FRACTION
    if candidate.makespan < current.makespan - tolerance:
        return True
    if candidate.makespan > current.makespan + tolerance:
        return False
    if candidate.deadhead < current.deadhead - EPS:
        return True
    if candidate.deadhead > current.deadhead + EPS:
        return False
    return candidate.overlap_area < current.overlap_area


def evaluate_assignment(name: str, geometry: FieldGeometry, machines: tuple[Machine, Machine],
                        passes: tuple[list[float], list[float]], overlap_area: float) -> Plan:
    """Оценить конкретное распределение проходов между двумя машинами.

    Для каждой машины перебираются сторона входа и порядок полос. Для каждой
    пары маршрутов ищется безопасная задержка старта, после чего выбирается
    лучший допустимый план.
    """

    options_1 = route_options(machines[0], geometry, passes[0])
    options_2 = route_options(machines[1], geometry, passes[1])
    if not options_1 or not options_2:
        return Plan(name=name, status="разворотная зона слишком узкая")

    best: Plan | None = None
    for route_1 in options_1:
        for route_2 in options_2:
            delays = find_safe_delays(machines[0], route_1, machines[1], route_2)
            if delays is None:
                continue
            delay_1, delay_2 = delays
            time_1 = delay_1 + route_1.duration
            time_2 = delay_2 + route_2.duration
            candidate = Plan(
                name=name,
                valid=True,
                status="OK",
                makespan=max(time_1, time_2),
                time_1=time_1,
                time_2=time_2,
                delay_1=delay_1,
                delay_2=delay_2,
                deadhead=route_1.deadhead_distance + route_2.deadhead_distance,
                overlap_area=overlap_area,
                turns=route_1.turn_count + route_2.turn_count,
                route_1=route_1,
                route_2=route_2,
            )
            if plan_is_better(candidate, best):
                best = candidate

    return best or Plan(name=name, status="столкновение не устраняется задержкой")


def region_passes(begin: float, width: float, header: float) -> list[float]:
    """Расположить целое число равномерных проходов внутри заданного участка."""

    if width <= EPS:
        return []
    count = max(1, math.ceil(width / header - EPS))
    spacing = width / count
    return [begin + (index + 0.5) * spacing for index in range(count)]


def evaluate_contiguous(name: str, geometry: FieldGeometry,
                        machines: tuple[Machine, Machine], share_1: float) -> Plan:
    """Оценить разделение поля на два смежных прямоугольных участка.

    ``share_1`` задаёт ширину участка первой машины. Дополнительно проверяется,
    с какого бокового края поля выгоднее расположить этот участок.
    """

    sweep = geometry.sweep_width
    widths = (share_1, sweep - share_1)
    best: Plan | None = None

    for first_machine_at_low_edge in (True, False):
        if first_machine_at_low_edge:
            begins = (0.0, widths[0])
        else:
            begins = (widths[1], 0.0)
        passes = (
            region_passes(begins[0], widths[0], machines[0].header),
            region_passes(begins[1], widths[1], machines[1].header),
        )
        overlap = (
            max(0.0, len(passes[0]) * machines[0].header - widths[0])
            + max(0.0, len(passes[1]) * machines[1].header - widths[1])
        ) * geometry.run_length
        candidate = evaluate_assignment(name, geometry, machines, passes, overlap)
        if plan_is_better(candidate, best):
            best = candidate

    return best or Plan(name=name)


def alternating_passes(geometry: FieldGeometry, machines: tuple[Machine, Machine],
                       first_machine: int) -> tuple[tuple[list[float], list[float]], float]:
    """Распределить последовательные проходы между машинами через один."""

    passes: tuple[list[float], list[float]] = ([], [])
    position = 0.0
    machine_index = first_machine
    overlap = 0.0
    while position < geometry.sweep_width - EPS:
        header = machines[machine_index].header
        effective = min(header, geometry.sweep_width - position)
        passes[machine_index].append(position + effective / 2.0)
        if effective + EPS < header:
            overlap += (header - effective) * 0.5 * geometry.run_length
        position += effective
        machine_index = 1 - machine_index
    return passes, overlap


def evaluate_strips(geometry: FieldGeometry,
                    machines: tuple[Machine, Machine]) -> Plan:
    """Оценить полосную стратегию для двух вариантов первой машины."""

    best: Plan | None = None
    for first_machine in (0, 1):
        passes, overlap = alternating_passes(geometry, machines, first_machine)
        candidate = evaluate_assignment("2_Полосами", geometry, machines, passes, overlap)
        if plan_is_better(candidate, best):
            best = candidate
    return best or Plan(name="2_Полосами")


def evaluate_strategies(geometry: FieldGeometry,
                        machines: tuple[Machine, Machine]) -> dict[str, Plan]:
    """Рассчитать три обязательные стратегии организации уборки.

    Возвращаются равное разделение, чередование полос и разделение
    пропорционально рабочей производительности ``ширина × скорость``.
    """

    sweep = geometry.sweep_width
    equal = evaluate_contiguous("1_Пополам", geometry, machines, sweep / 2.0)
    capacity_1 = machines[0].header * machines[0].work_speed
    capacity_2 = machines[1].header * machines[1].work_speed
    productivity_share = sweep * capacity_1 / (capacity_1 + capacity_2)
    productivity = evaluate_contiguous(
        "3_По_производительности", geometry, machines, productivity_share
    )
    strips = evaluate_strips(geometry, machines)
    return {plan.name: plan for plan in (equal, strips, productivity)}


def make_machine(header: float, speed: float, length: float, width: float,
                 start: Point, orientation_degrees: float) -> Machine:
    """Создать описание машины и вывести дополнительные скорости и радиус."""

    return Machine(
        header=header,
        work_speed=speed,
        travel_speed=speed * TRAVEL_SPEED_FACTOR,
        turn_speed=speed * TURN_SPEED_FACTOR,
        length=length,
        width=width,
        turn_radius=max(MINIMUM_TURN_RADIUS, length / 2.0, width / 2.0),
        start=start,
        orientation=math.radians(orientation_degrees),
    )


def condition_values(condition: tuple[str, str, str, str]) -> tuple[float, ...]:
    """Преобразовать отношения ``=``, ``>`` и ``<`` в параметры двух машин."""

    header_condition, speed_condition, length_condition, width_condition = condition

    def pair(values: dict[str, float], relation: str) -> tuple[float, float]:
        """Получить пару значений, где вторая машина является базовой."""

        return (values[relation], values["="]) if relation != "=" else (values["="], values["="])

    header_1, header_2 = pair(HEADER_VARIANTS, header_condition)
    speed_1, speed_2 = pair(SPEED_VARIANTS, speed_condition)
    length_1, length_2 = pair(LENGTH_VARIANTS, length_condition)
    width_1, width_2 = pair(WIDTH_VARIANTS, width_condition)
    return header_1, header_2, speed_1, speed_2, length_1, length_2, width_1, width_2


def generate_experiment() -> list[dict[str, object]]:
    """Сформировать и рассчитать полный эксперимент из 162 сценариев.

    Для двух направлений проходов перебираются по три отношения ширины захвата,
    скорости, длины и ширины машин: ``2 × 3⁴ = 162`` сценария.
    """

    rows: list[dict[str, object]] = []
    relations: tuple[str, str, str] = ("=", ">", "<")
    conditions: list[tuple[str, str, str, str]] = [
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

            machines: tuple[Machine, Machine] = (
                make_machine(header_1, speed_1, length_1, width_1,
                             start_1, orientation_1),
                make_machine(header_2, speed_2, length_2, width_2,
                             start_2, orientation_2),
            )
            geometry = FieldGeometry(axis)
            plans = evaluate_strategies(geometry, machines)
            valid_plans: list[Plan] = [plan for plan in plans.values() if plan.valid]
            winner: Plan | None = None
            for plan in valid_plans:
                if plan_is_better(plan, winner):
                    winner = plan

            gap_tolerance = max(
                (sweep_width % header_1) / sweep_width,
                (sweep_width % header_2) / sweep_width,
            )

            def value(name: str, attribute: str, digits: int = 1) -> float | None:
                """Безопасно получить округлённый числовой показатель плана."""

                plan = plans[name]
                if not plan.valid:
                    return None
                raw_value: object = getattr(plan, attribute)
                if not isinstance(raw_value, (int, float)):
                    raise TypeError(f"Атрибут {attribute} не является числом")
                return round(float(raw_value), digits)

            rows.append(
                {
                    "s": axis,
                    "cond_l": header_relation,
                    "cond_U": speed_relation,
                    "cond_hx": length_relation,
                    "cond_hy": width_relation,
                    "l1": header_1,
                    "l2": header_2,
                    "U1": speed_1,
                    "U2": speed_2,
                    "Vtravel1": machines[0].travel_speed,
                    "Vtravel2": machines[1].travel_speed,
                    "hx1": length_1,
                    "hy1": width_1,
                    "hx2": length_2,
                    "hy2": width_2,
                    "R1": machines[0].turn_radius,
                    "R2": machines[1].turn_radius,
                    "coord1_x": start_1[0],
                    "coord1_y": start_1[1],
                    "coord2_x": start_2[0],
                    "coord2_y": start_2[1],
                    "orient1": orientation_1,
                    "orient2": orientation_2,
                    "propusk_%": round(gap_tolerance * 100.0, 2),
                    "T_strat1": value("1_Пополам", "makespan"),
                    "T_strat2": value("2_Полосами", "makespan"),
                    "T_strat3": value("3_По_производительности", "makespan"),
                    "turns_strat1": value("1_Пополам", "turns", 0),
                    "turns_strat2": value("2_Полосами", "turns", 0),
                    "turns_strat3": value("3_По_производительности", "turns", 0),
                    "delay_strat1": value("1_Пополам", "delay_1", 0)
                    or value("1_Пополам", "delay_2", 0),
                    "delay_strat2": value("2_Полосами", "delay_1", 0)
                    or value("2_Полосами", "delay_2", 0),
                    "delay_strat3": value("3_По_производительности", "delay_1", 0)
                    or value("3_По_производительности", "delay_2", 0),
                    "status_strat1": plans["1_Пополам"].status,
                    "status_strat2": plans["2_Полосами"].status,
                    "status_strat3": plans["3_По_производительности"].status,
                    "Winner": winner.name if winner else "Нет допустимой стратегии",
                    "Winner_time": round(winner.makespan, 1) if winner else None,
                }
            )

    return rows


def save_results(rows: list[dict[str, object]]) -> Path:
    """Сохранить результаты в XLSX, а без openpyxl — в UTF-8 CSV.

    Returns:
        Фактический путь созданного файла.
    """

    frame = pd.DataFrame(rows)
    excel_path = Path("synthetic_data_experiment_v2.xlsx")
    try:
        frame.to_excel(excel_path, index=False)  # pyright: ignore[reportUnknownMemberType]
        return excel_path
    except ModuleNotFoundError as error:
        if error.name != "openpyxl":
            raise
        csv_path = excel_path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path


def main() -> None:
    """Запустить эксперимент, сохранить таблицу и вывести краткую статистику."""

    rows = generate_experiment()
    output_path = save_results(rows)
    frame = pd.DataFrame(rows)

    print(f"Обработано сценариев: {len(frame)}")
    print(f"Результат сохранён: {output_path}")
    if output_path.suffix == ".csv":
        print("Для XLSX установите openpyxl: python -m pip install openpyxl")
    print("\nПервые пять результатов:")
    print(frame[["s", "l1", "l2", "U1", "U2", "T_strat1", "T_strat2",
                 "T_strat3", "Winner"]].head())
    print("\nПобедители:")
    print(frame["Winner"].value_counts(dropna=False))
    print("\nНедопустимые стратегии:")
    for column in ("status_strat1", "status_strat2", "status_strat3"):
        print(column, (frame[column] != "OK").sum())


if __name__ == "__main__":
    main()
