"""Two OFFLINE HW4 planners: conservative-AABB visibility and max-clearance grid.

No ROS imports. No geometry library. Never publishes motor commands.
"""
from __future__ import annotations

import heapq
import math

from geometry import (EPS, expanded_rect, norm, point_rect_distance,
                      segment_clearances, segment_crosses_open_rect, wall_clearance)


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
    # Two routes can have the same Euclidean length in this symmetric arena.
    # Break *distance ties only* using the rover's measured initial heading,
    # avoiding a gratuitous near-180-degree start turn. This does not change
    # the primary shortest-distance objective.
    start_yaw = math.radians(float(world.cfg.get('start_yaw_deg', 0.0)))
    best = [(float('inf'), float('inf')) for _ in nodes]
    prev = {}
    best[0] = (0.0, 0.0)
    heap = [(0.0, 0.0, 0)]
    while heap:
        dist, initial_turn, u = heapq.heappop(heap)
        if (dist > best[u][0] + EPS or
            (abs(dist - best[u][0]) <= EPS and initial_turn > best[u][1] + EPS)):
            continue
        if u == 1:
            break
        for v, edge_len in graph[u]:
            candidate_dist = dist + edge_len
            candidate_turn = initial_turn
            if u == 0:
                bearing = math.atan2(nodes[v][1] - nodes[0][1],
                                     nodes[v][0] - nodes[0][0])
                candidate_turn = abs(math.atan2(math.sin(bearing - start_yaw),
                                                math.cos(bearing - start_yaw)))
            old_dist, old_turn = best[v]
            if (candidate_dist < old_dist - EPS or
                (abs(candidate_dist - old_dist) <= EPS and candidate_turn < old_turn - EPS)):
                best[v] = (candidate_dist, candidate_turn)
                prev[v] = u
                heapq.heappush(heap, (candidate_dist, candidate_turn, v))
    distance = [x[0] for x in best]
    if math.isinf(distance[1]):
        raise ValueError('No visibility-graph path')
    path = [1]
    while path[-1] != 0:
        path.append(prev[path[-1]])
    path.reverse()
    return [nodes[i] for i in path], {'nodes': nodes, 'edges': visible_edges}


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
