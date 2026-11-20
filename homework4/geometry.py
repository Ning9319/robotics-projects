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
