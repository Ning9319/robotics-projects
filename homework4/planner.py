"""HW4 2D geometry written from first principles (no Shapely/geometry library).

Conservative C-space: a circular robot with radius r is treated as a point,
but the rectangular obstacle is expanded to an AXIS-ALIGNED rectangle by r.
This slightly over-inflates the true rounded-corner Minkowski sum.
"""
from __future__ import annotations

import math

EPS = 1e-9


def norm(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_segment_distance(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    vv = vx * vx + vy * vy
    if vv < EPS:
        return norm(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / vv))
    return math.hypot(p[0] - a[0] - t * vx, p[1] - a[1] - t * vy)


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(p, a, b):
    return (abs(cross(a, b, p)) <= EPS
            and min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
            and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS)


def segments_intersect(a, b, c, d):
    o1, o2 = cross(a, b, c), cross(a, b, d)
    o3, o4 = cross(c, d, a), cross(c, d, b)
    if (o1 > EPS and o2 < -EPS or o1 < -EPS and o2 > EPS) and (
        o3 > EPS and o4 < -EPS or o3 < -EPS and o4 > EPS
    ):
        return True
    return ((abs(o1) <= EPS and on_segment(c, a, b)) or
            (abs(o2) <= EPS and on_segment(d, a, b)) or
            (abs(o3) <= EPS and on_segment(a, c, d)) or
            (abs(o4) <= EPS and on_segment(b, c, d)))


def segment_segment_distance(a, b, c, d):
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def point_rect_distance(p, rect):
    x0, y0, x1, y1 = rect
    return math.hypot(max(x0 - p[0], 0.0, p[0] - x1),
                      max(y0 - p[1], 0.0, p[1] - y1))


def rect_edges(rect):
    x0, y0, x1, y1 = rect
    q = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(q[i], q[(i + 1) % 4]) for i in range(4)]


def segment_rect_distance(a, b, rect):
    if point_rect_distance(a, rect) < EPS or point_rect_distance(b, rect) < EPS:
        return 0.0
    return min(segment_segment_distance(a, b, c, d) for c, d in rect_edges(rect))


def expanded_rect(rect, delta):
    return rect[0] - delta, rect[1] - delta, rect[2] + delta, rect[3] + delta


def segment_crosses_open_rect(a, b, rect):
    """True iff any portion of segment is strictly INSIDE rectangle.

    Contact with the inflated boundary/corners is permitted for visibility paths.
    Collect crossing parameters, test the middle of each resulting interval.
    """
    x0, y0, x1, y1 = rect
    ts = [0.0, 1.0]
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) > EPS:
        ts.extend((x - a[0]) / dx for x in (x0, x1))
    if abs(dy) > EPS:
        ts.extend((y - a[1]) / dy for y in (y0, y1))
    ts = sorted({max(0.0, min(1.0, t)) for t in ts if -EPS <= t <= 1 + EPS})
    for lo, hi in zip(ts, ts[1:]):
        t = (lo + hi) / 2
        x, y = a[0] + t * dx, a[1] + t * dy
        if x0 + EPS < x < x1 - EPS and y0 + EPS < y < y1 - EPS:
            return True
    return False


def wall_clearance(p, side):
    return min(p[0], p[1], side - p[0], side - p[1])


def segment_clearances(a, b, side, raw_obstacle):
    # Minimum to walls occurs at endpoints because wall distance is affine.
    walls = min(wall_clearance(a, side), wall_clearance(b, side))
    obstacle = segment_rect_distance(a, b, raw_obstacle)
    return obstacle, walls


def path_metrics(points, side, raw_obstacle):
    length = sum(norm(a, b) for a, b in zip(points, points[1:]))
    obstacle_clearance = float('inf')
    wall_clearance_m = float('inf')
    for a, b in zip(points, points[1:]):
        d_obs, d_wall = segment_clearances(a, b, side, raw_obstacle)
        obstacle_clearance = min(obstacle_clearance, d_obs)
        wall_clearance_m = min(wall_clearance_m, d_wall)
    return {
        'length_m': round(length, 5),
        'min_obstacle_clearance_m': round(obstacle_clearance, 5),
        'min_wall_clearance_m': round(wall_clearance_m, 5),
        'min_any_clearance_m': round(min(obstacle_clearance, wall_clearance_m), 5),
        'waypoints': len(points),
    }


"""Two OFFLINE HW4 planners: conservative-AABB visibility and max-clearance grid.

No ROS imports. No geometry library. Never publishes motor commands.
"""

import heapq
import math



class World:
    def __init__(self, cfg):
        self.cfg = cfg
        self.side = float(cfg['workspace']['side_m'])
        self.start = tuple(map(float, cfg['start_xy_m']))
        self.goal = tuple(map(float, cfg['goal_xy_m']))
        o = cfg['obstacle']
        cx, cy = map(float, o['center_xy_m'])
        w, h = map(float, o['size_xy_m'])
        self.raw_obstacle = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        self.robot_radius = float(cfg['robot']['radius_m'])
        self.base_margin = float(cfg['planning']['base_margin_m'])
        self.inflation = self.robot_radius + self.base_margin
        self.inflated = expanded_rect(self.raw_obstacle, self.inflation)
        self.grid_step = float(cfg['planning']['grid_step_m'])
        self.tags = {str(k): (float(v['x']), float(v['y']))
                     for k, v in cfg.get('landmarks', {}).items()}
        if self.side <= 0 or w <= 0 or h <= 0 or self.inflation <= 0 or self.grid_step <= 0:
            raise ValueError('Invalid workspace, obstacle, robot radius, or grid step')
        if not (0 < self.raw_obstacle[0] < self.raw_obstacle[2] < self.side and
                0 < self.raw_obstacle[1] < self.raw_obstacle[3] < self.side):
            raise ValueError('Obstacle is outside the workspace')
        if not (0.3048 <= w <= 0.6096 and 0.3048 <= h <= 0.6096):
            raise ValueError('HW4 obstacle sides must be within 1–2 feet')
        for name, p in [('start', self.start), ('goal', self.goal)]:
            if wall_clearance(p, self.side) < self.inflation - EPS:
                raise ValueError(f'{name} is too close to a wall for the robot')
            if segment_crosses_open_rect(p, p, self.inflated):
                raise ValueError(f'{name} is inside inflated obstacle')

    def point_free(self, p):
        return (wall_clearance(p, self.side) >= self.inflation - EPS
                and not segment_crosses_open_rect(p, p, self.inflated))

    def segment_free(self, a, b):
        return (wall_clearance(a, self.side) >= self.inflation - EPS
                and wall_clearance(b, self.side) >= self.inflation - EPS
                and not segment_crosses_open_rect(a, b, self.inflated))

    def segment_clearance(self, a, b):
        return min(segment_clearances(a, b, self.side, self.raw_obstacle))


def shortest_visibility(world):
    """Euclidean Dijkstra shortest path on a rectangle-corner visibility graph.

    Exact for THIS conservative axis-aligned rectangular C-space model;
    not necessarily globally shortest for the true circular-arc inflated shape.
    """
    x0, y0, x1, y1 = world.inflated
    nodes = [world.start, world.goal,
             (x0, y0), (x0, y1), (x1, y0), (x1, y1)]
    graph = [[] for _ in nodes]
    visible_edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if world.segment_free(nodes[i], nodes[j]):
                length = norm(nodes[i], nodes[j])
                graph[i].append((j, length))
                graph[j].append((i, length))
                visible_edges.append((nodes[i], nodes[j]))
    distance = [float('inf')] * len(nodes)
    prev = {}
    distance[0] = 0.0
    heap = [(0.0, 0)]
    while heap:
        dist, u = heapq.heappop(heap)
        if dist > distance[u] + EPS:
            continue
        if u == 1:
            break
        for v, edge_len in graph[u]:
            cand = dist + edge_len
            if cand + EPS < distance[v]:
                distance[v], prev[v] = cand, u
                heapq.heappush(heap, (cand, v))
    if math.isinf(distance[1]):
        raise ValueError('No visibility-graph path')
    path = [1]
    while path[-1] != 0:
        path.append(prev[path[-1]])
    path.reverse()
    result = [nodes[i] for i in path]

    # Equal-distance tie: prefer the same side as the Safety path.
    preferred = [world.start, (x1, y1), world.goal]
    preferred_len = sum(norm(a, b) for a, b in zip(preferred, preferred[1:]))
    if (abs(preferred_len - distance[1]) < 1e-8 and
            all(world.segment_free(a, b)
                for a, b in zip(preferred, preferred[1:]))):
        result = preferred

    return result, {'nodes': nodes, 'edges': visible_edges}


def max_clearance_grid(world):
    """Widest-path search (maximize min clearance), tie-break by shorter length.

    Grid resolution limits optimality. Edge clearance checks use the entire
    segment, not just its endpoints (prevents diagonal corner cutting).
    """
    step = world.grid_step
    n = int(math.floor(world.side / step))
    lattice = {}
    nodes = [world.start, world.goal]
    for ix in range(n + 1):
        for iy in range(n + 1):
            pt = (round(ix * step, 8), round(iy * step, 8))
            if world.point_free(pt):
                lattice[(ix, iy)] = len(nodes)
                nodes.append(pt)
    edges = [[] for _ in nodes]
    for (ix, iy), i in lattice.items():
        for dx, dy in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
            j = lattice.get((ix + dx, iy + dy))
            if j is not None and world.segment_free(nodes[i], nodes[j]):
                length = norm(nodes[i], nodes[j])
                clearance = world.segment_clearance(nodes[i], nodes[j])
                edges[i].append((j, length, clearance))
                edges[j].append((i, length, clearance))
    # The arbitrary off-grid start/goal connect to nearby *valid* grid cells.
    for special in (0, 1):
        near = sorted((norm(nodes[special], pt), i)
                      for i, pt in enumerate(nodes) if i >= 2)
        linked = 0
        for d, i in near:
            if d > 2.2 * step and linked:
                break
            if world.segment_free(nodes[special], nodes[i]):
                clearance = world.segment_clearance(nodes[special], nodes[i])
                edges[special].append((i, d, clearance))
                edges[i].append((special, d, clearance))
                linked += 1
            if linked >= 12:
                break
        if not linked:
            raise ValueError('Start/goal cannot connect to grid; reduce grid step')
    # Heap maximizes bottleneck first, then minimizes path length at that bottleneck.
    best = [(-1.0, float('inf')) for _ in nodes]
    best[0] = (min(wall_clearance(world.start, world.side),
                   point_rect_distance(world.start, world.raw_obstacle)), 0.0)
    prev = {}
    heap = [(-best[0][0], 0.0, 0)]
    while heap:
        neg_c, length, u = heapq.heappop(heap)
        clearance = -neg_c
        if clearance < best[u][0] - EPS or (
            abs(clearance - best[u][0]) <= EPS and length > best[u][1] + EPS
        ):
            continue
        if u == 1:
            break
        for v, d, edge_clearance in edges[u]:
            bottleneck = min(clearance, edge_clearance)
            new_length = length + d
            old_c, old_length = best[v]
            if (bottleneck > old_c + EPS or
                    (abs(bottleneck - old_c) <= EPS and new_length + EPS < old_length)):
                best[v] = (bottleneck, new_length)
                prev[v] = u
                heapq.heappush(heap, (-bottleneck, new_length, v))
    if 1 not in prev:
        raise ValueError('No safe grid path; check inflated obstacle and map')
    path = [1]
    while path[-1] != 0:
        path.append(prev[path[-1]])
    path.reverse()
    polyline = [nodes[i] for i in path]
    # Shorten the grid staircase WITHOUT reducing its proven bottleneck.
    # Unlike unconstrained shortcutting, every simplified segment must retain
    # the max-clearance objective.
    threshold = best[1][0] - 1e-6
    simplified = [polyline[0]]
    i = 0
    while i < len(polyline) - 1:
        j = len(polyline) - 1
        while j > i + 1:
            if world.segment_free(polyline[i], polyline[j]) and (
                world.segment_clearance(polyline[i], polyline[j]) >= threshold
            ):
                break
            j -= 1
        simplified.append(polyline[j])
        i = j
    return simplified, {'grid_count': len(lattice),
                        'grid_path_count': len(polyline),
                        'optimal_grid_bottleneck_m': best[1][0],
                        'grid_samples': [nodes[i] for i in lattice.values()]}


def validate_path(world, path):
    failures = []
    for i, (a, b) in enumerate(zip(path, path[1:])):
        if not world.segment_free(a, b):
            failures.append(i)
    return failures
