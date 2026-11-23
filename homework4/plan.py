#!/usr/bin/env python3
"""HW4 offline-only planning and visualization. NEVER controls the robot.

Usage:
  python3 homework4/plan.py --config homework4/config.yaml
  python3 homework4/plan.py --config homework4/config.yaml --out homework4/output/zero_margin
"""
import argparse
import json
from pathlib import Path

import yaml

from geometry import path_metrics
from planning import World, max_clearance_grid, shortest_visibility, validate_path


def rect_patch(ax, rect, **kwargs):
    from matplotlib.patches import Rectangle
    x0, y0, x1, y1 = rect
    patch = Rectangle((x0, y0), x1 - x0, y1 - y0, **kwargs)
    ax.add_patch(patch)
    return patch


def draw_base(ax, world):
    ax.set_xlim(-0.15, world.side + 0.15)
    ax.set_ylim(-0.15, world.side + 0.15)
    ax.set_aspect('equal')
    ax.set_xlabel('World x (m)')
    ax.set_ylabel('World y (m)')
    ax.grid(alpha=0.12)
    rect_patch(ax, (0, 0, world.side, world.side), fill=False, linewidth=2, edgecolor='black')
    rect_patch(ax, world.raw_obstacle, facecolor='#f5b8ad', edgecolor='#b73b37', label='Raw obstacle')
    rect_patch(ax, world.inflated, fill=False, linestyle='--', linewidth=1.7,
               edgecolor='#b73b37', label='Robot-radius inflated obstacle')
    for tag, (x, y) in world.tags.items():
        ax.scatter([x], [y], marker='s', color='#555', s=25)
        ax.annotate(f'T{tag}', (x, y), xytext=(4, 3), textcoords='offset points', fontsize=7)
    ax.scatter([world.start[0]], [world.start[1]], marker='o', s=90, color='#1a8b5b', zorder=7)
    ax.scatter([world.goal[0]], [world.goal[1]], marker='*', s=120, color='#15699b', zorder=7)
    ax.annotate('START', world.start, xytext=(4, -16), textcoords='offset points', fontsize=8, weight='bold')
    ax.annotate('GOAL', world.goal, xytext=(4, 7), textcoords='offset points', fontsize=8, weight='bold')


def draw_path(ax, path, color, name):
    xx, yy = zip(*path)
    ax.plot(xx, yy, color=color, marker='o', markersize=3.5,
            linewidth=2.1, zorder=6, label=name)


def make_plots(world, shortest, vis, safest, safety_info, stem, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=140)
    draw_base(ax, world)
    for a, b in vis['edges']:
        ax.plot((a[0], b[0]), (a[1], b[1]), color='#bec6cb', linewidth=0.8, zorder=1)
    for x, y in vis['nodes'][2:]:
        ax.scatter([x], [y], c='#7f8d96', s=19)
    draw_path(ax, shortest, '#104c8a', 'Shortest path (visibility + Dijkstra)')
    ax.set_title('HW4 — Minimum-distance plan | ' + mode, fontsize=12)
    ax.legend(loc='lower left', fontsize=7, framealpha=.95)
    fig.tight_layout()
    fig.savefig(str(stem) + '_shortest.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=140)
    # Clearance field for eligible grid cells. This is the safety representation.
    samples = safety_info['grid_samples']
    if samples:
        c = [min(x, y, world.side - x, world.side - y,
                 math_point_rect_distance((x, y), world.raw_obstacle)) for x, y in samples]
        sc = ax.scatter([p[0] for p in samples], [p[1] for p in samples], c=c,
                        cmap='viridis', norm=Normalize(vmin=0, vmax=max(c)),
                        s=24, marker='s', linewidth=0, zorder=0)
        fig.colorbar(sc, ax=ax, shrink=0.78, label='Point clearance to raw obstacle/walls (m)')
    draw_base(ax, world)
    draw_path(ax, safest, '#e58415', 'Maximum-clearance path (grid widest path)')
    ax.set_title('HW4 — Maximum-safety plan | ' + mode, fontsize=12)
    ax.legend(loc='lower left', fontsize=7, framealpha=.95)
    fig.tight_layout()
    fig.savefig(str(stem) + '_safety.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=140)
    draw_base(ax, world)
    draw_path(ax, safest, '#e58415', 'Safety')
    draw_path(ax, shortest, '#104c8a', 'Distance')
    ax.set_title('HW4 — Planned-path trade-off | ' + mode, fontsize=12)
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(str(stem) + '_comparison.png', dpi=200)
    plt.close(fig)


def math_point_rect_distance(p, rect):
    from geometry import point_rect_distance
    return point_rect_distance(p, rect)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='homework4/config.yaml', help='Measured HW4 map YAML')
    parser.add_argument('--out', default=None, help='Output prefix (default: homework4/output/<config-stem>)')
    args = parser.parse_args()
    conf_path = Path(args.config)
    cfg = yaml.safe_load(conf_path.read_text(encoding='utf-8'))
    # Fail clearly if the real HW4 measurements have not been filled in.
    required = {
        'start_xy_m': cfg.get('start_xy_m'),
        'goal_xy_m': cfg.get('goal_xy_m'),
        'obstacle.center_xy_m': cfg.get('obstacle', {}).get('center_xy_m'),
        'obstacle.size_xy_m': cfg.get('obstacle', {}).get('size_xy_m'),
        'robot.radius_m': cfg.get('robot', {}).get('radius_m'),
        'planning.base_margin_m': cfg.get('planning', {}).get('base_margin_m'),
    }
    missing = [k for k, val in required.items() if val is None or
               (isinstance(val, list) and any(x is None for x in val))]
    if missing:
        parser.error('Site measurements still missing: ' + ', '.join(missing))
    world = World(cfg)
    diagonals = [
        max(abs(world.start[0] - world.start[1]), abs(world.goal[0] - world.goal[1])),
        max(abs(sum(world.start) - world.side), abs(sum(world.goal) - world.side)),
    ]
    if min(diagonals) > 0.08:
        print('WARNING: start/goal do not appear to be on the same world diagonal; check HW4 handout.')
    mode = str(cfg.get('data_status', 'UNSPECIFIED'))
    if mode != 'MEASURED_HW4_VERIFIED':
        print('*** MEASURED MAP; APRILTAG POSE AND PHYSICAL CLEARANCE NOT YET ROBOT-VERIFIED ***')
    shortest, visibility = shortest_visibility(world)
    safest, safety_info = max_clearance_grid(world)
    for name, points in [('shortest', shortest), ('safety', safest)]:
        invalid = validate_path(world, points)
        if invalid:
            raise AssertionError(f'{name} path has colliding segments: {invalid}')

    out = Path(args.out) if args.out else Path('homework4/output') / conf_path.stem
    out.parent.mkdir(parents=True, exist_ok=True)
    methods = {'shortest': shortest, 'safety': safest}
    report = {
        'source_config': str(conf_path), 'data_status': mode,
        'workspace_m': world.side,
        'conservative_inflation_m': world.inflation,
        'NOTE': 'These are PLANNED paths, not actual executed rover trajectories.',
        'paths': {
            name: {'waypoints_xy_m': [list(p) for p in route],
                   'metrics': path_metrics(route, world.side, world.raw_obstacle),
                   'collision_free_in_conservative_cspace': True}
            for name, route in methods.items()
        },
        'safety_grid': {k: v for k, v in safety_info.items() if k != 'grid_samples'},
    }
    (out.parent / (out.name + '_plans.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    make_plots(world, shortest, visibility, safest, safety_info, out, mode)
    for name in ['shortest', 'safety']:
        print(name.upper(), report['paths'][name]['metrics'])
    print('Saved:', *(str(out) + '_' + t for t in
                      ['shortest.png', 'safety.png', 'comparison.png', 'plans.json']), sep='\n  ')


if __name__ == '__main__':
    main()
