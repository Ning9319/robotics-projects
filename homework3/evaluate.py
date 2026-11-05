#!/usr/bin/env python3
"""HW3 evaluation: SLAM map vs. tape-measured ground truth (runs on a laptop, no ROS).

    python3 homework3/evaluate.py                                  # square_final + octagon_final
    python3 homework3/evaluate.py square_run2 octagon_run4         # any runs in output/runs/

The SLAM map lives in the robot's own start frame. Following handout item 4, it
is moved into the world frame with the MEASURED start pose
(config/ground_truth.yaml -> start_poses):  p_world = R(yaw0) p_slam + (x0, y0).
Covariances are rotated the same way. Errors are per matched tag ID.

Also reported, as analysis only (never used to change the estimate):
  * shape error: mean error after the best rigid fit (start-pose error removed),
  * scale: best similarity-fit scale (systematic range error shows up here).

Outputs (homework3/output/figures/ and homework3/output/):
  ground_truth.png, map_<run>.png, laps.png, nis.png, results.md, results.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'homework3/output/runs'
FIG = ROOT / 'homework3/output/figures'
CHI2_2DOF_95 = 5.991

# Reference categorical palette (validated: all-pairs CVD dE >= 9.2, normal >= 24).
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
SHAPE_COLOR = {'square': BLUE, 'octagon': ORANGE}


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def load_run(name, gt):
    d = RUNS / name
    cfg = json.loads((d / 'config.json').read_text())
    shape = cfg['args']['path']
    sp = gt['start_poses'][shape]
    if sp['x'] is None:
        raise SystemExit(f'start pose for {shape} missing in ground_truth.yaml')
    start = (float(sp['x']), float(sp['y']), math.radians(float(sp['yaw_deg'])))
    snaps = [s for s in json.loads((d / 'snapshots.json').read_text())
             if str(s['reason']).startswith('end of lap')]
    with open(d / 'trajectory.csv', newline='', encoding='utf-8') as f:
        traj = list(csv.DictReader(f))
    with open(d / 'updates.csv', newline='', encoding='utf-8') as f:
        nis = [float(r['nis']) for r in csv.DictReader(f) if r['nis']]
    return {'name': name, 'cfg': cfg, 'shape': shape, 'start': start, 'snaps': snaps,
            'traj': traj, 'nis': np.array(nis), 'side': cfg['side_m'], 'laps': cfg['args']['laps']}


def to_world(start, p):
    return (rot(start[2]) @ np.asarray(p).T).T + start[:2]


def landmarks_world(snap, start):
    x, P, idx = np.array(snap['x']), np.array(snap['P']), snap['landmark_index']
    R = rot(start[2])
    out = {}
    for tag, j in idx.items():
        m = to_world(start, x[j:j + 2][None])[0]
        out[int(tag)] = (m, R @ P[j:j + 2, j:j + 2] @ R.T)
    return out


def evaluate_snapshot(snap, run, gt_tags):
    lms = landmarks_world(snap, run['start'])
    ids = sorted(t for t in lms if t in gt_tags)
    est = np.array([lms[t][0] for t in ids])
    true = np.array([gt_tags[t] for t in ids])
    err = np.linalg.norm(est - true, axis=1)
    A, B = est - est.mean(0), true - true.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    Rf = (U @ D @ Vt).T
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum()
    shape_err = np.linalg.norm((Rf @ A.T).T - B, axis=1)
    sigma = np.array([math.sqrt(np.trace(lms[t][1])) for t in ids])   # sqrt(var_x + var_y)
    return {'ids': ids, 'est': est, 'true': true, 'err': err, 'shape_err': shape_err,
            'scale_pct': 100 * (1 / s - 1), 'sigma': sigma, 'lms': lms,
            'mapped': len(lms), 'missed': sorted(set(gt_tags) - set(lms)),
            'extra': sorted(set(lms) - set(gt_tags))}


def point_to_polyline(p, poly):
    """Shortest distance from point p to a closed polyline (list of vertices)."""
    best = float('inf')
    for a, b in zip(poly, poly[1:]):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def path_metrics(run):
    """Path accuracy in the SLAM frame (independent of the measured start pose).

    Uses the EKF pose at the end of every stop (after the camera correction),
    i.e. the best estimate of where the robot was. cross-track = distance from
    that pose to the planned polygon; vertex error = distance to the planned
    vertex when the controller declared it reached; std = filter's own robot
    position std at the same instants (how far the estimate itself can be off).
    """
    k = 4 if run['shape'] == 'square' else 8
    poly = np.array([(0.0, 0.0)] + [tuple(w[:2]) for w in run['cfg']['waypoints'][:k]])
    rows = run['traj']
    out = {}
    for prev, row in zip(rows, rows[1:]):
        if row['phase'] == 'PLAN' and prev['phase'] in ('OBSERVE', 'HOLD') and row['lap']:
            lap = int(row['lap'])
            p = np.array([float(row['x_m']), float(row['y_m'])])
            d = out.setdefault(lap, {'xtrack': [], 'std': []})
            d['xtrack'].append(point_to_polyline(p, poly))
            d['std'].append(math.sqrt(float(row['var_xx']) + float(row['var_yy'])))
    with open(RUNS / run['name'] / 'vertices.csv', newline='', encoding='utf-8') as f:
        for v in csv.DictReader(f):
            out.setdefault(int(v['lap']), {'xtrack': [], 'std': []}).setdefault('vertex', []).append(
                float(v['dist_to_target_m']))
    return {lap: {key: np.array(val) for key, val in d.items()} for lap, d in out.items()}


def ellipse(mean, cov, k=math.sqrt(CHI2_2DOF_95), n=72):
    w, v = np.linalg.eigh(cov)
    w = np.sqrt(np.maximum(w, 0)) * k
    t = np.linspace(0, 2 * math.pi, n)
    return mean[:, None] + v @ (w[:, None] * np.vstack([np.cos(t), np.sin(t)]))


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def draw_field(ax, gt, label_tags=True):
    L = gt['field_side_m']
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color=INK2, lw=1.2)
    for t, (x, y) in gt['tag_xy'].items():
        ax.plot(x, y, marker='x', ms=8, mew=2, color=INK, ls='none')
        if label_tags:   # label just OUTSIDE the field, away from marks and arrows
            dx = -0.08 if x < 0.01 else (0.08 if x > L - 0.01 else 0)
            dy = -0.08 if y < 0.01 else (0.08 if y > L - 0.01 else 0)
            ax.text(x + dx, y + dy, str(t), color=INK, fontsize=9, ha='center', va='center',
                    fontweight='bold')


def planned_world(run):
    pts = [(0.0, 0.0)] + [tuple(w[:2]) for w in run['cfg']['waypoints']]
    k = 4 if run['shape'] == 'square' else 8
    return to_world(run['start'], np.array(pts[:k + 1]))


def fig_ground_truth(gt, runs, plt):
    fig, ax = plt.subplots(figsize=(6, 6))
    style(ax)
    draw_field(ax, gt)
    L = gt['field_side_m']
    for t, (x, y) in gt['tag_xy'].items():   # facing arrows (into the field)
        nx = 1 if x < 0.01 else (-1 if x > L - 0.01 else 0)
        ny = 1 if y < 0.01 else (-1 if y > L - 0.01 else 0)
        ax.annotate('', (x + 0.16 * nx, y + 0.16 * ny), (x, y),
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=1))
    seen = set()
    for run in runs:
        if run['shape'] in seen:
            continue
        seen.add(run['shape'])
        p = planned_world(run)
        c = SHAPE_COLOR[run['shape']]
        ax.plot(p[:, 0], p[:, 1], color=c, lw=2, ls='--')
        x0, y0, th = run['start']
        ax.annotate('', (x0 + 0.18 * math.cos(th), y0 + 0.18 * math.sin(th)), (x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=2))
        ax.plot(x0, y0, 'o', color=c, ms=8, mec='white', mew=2)
        ax.text(p[:, 0].mean(), p[:, 1].mean() + (0.08 if run['shape'] == 'square' else -0.08),
                f'{run["shape"]} ({run["side"]:.1f} m side)', color=INK2, fontsize=9, ha='center')
    ax.set_aspect('equal')
    ax.set_xlim(-0.2, L + 0.2)
    ax.set_ylim(-0.2, L + 0.2)
    ax.set_xlabel('x (m)', color=INK2)
    ax.set_ylabel('y (m)', color=INK2)
    ax.set_title(f'Ground truth: {L:.2f} m field, 8 AprilTags (x), planned paths and start poses',
                 color=INK, fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(FIG / 'ground_truth.png', dpi=160)
    plt.close(fig)


def fig_map(run, res, gt, plt):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 5.8))
    gs = GridSpec(2, 6, figure=fig, width_ratios=[1.3, 1.3, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[:, :2])
    style(ax)
    draw_field(ax, gt)
    p = planned_world(run)
    ax.plot(p[:, 0], p[:, 1], color=MUTED, lw=1, ls='--', label='planned path')
    xy = to_world(run['start'], np.array([[float(r['x_m']), float(r['y_m'])] for r in run['traj']]))
    ax.plot(xy[:, 0], xy[:, 1], color=BLUE, lw=1.5, label='estimated robot trajectory (EKF)')
    for t, (m, C) in res['lms'].items():
        e = ellipse(m, C)
        ax.plot(e[0], e[1], color=AQUA, lw=1.5)
        ax.plot(m[0], m[1], 'o', color=AQUA, ms=5, mec='white', mew=1)
    ax.plot([], [], 'o-', color=AQUA, lw=1.5, ms=5, label='estimated landmark + 95% ellipse')
    ax.plot([], [], 'x', color=INK, ms=8, mew=2, label='ground-truth tag')
    ax.plot(*run['start'][:2], 'o', color=BLUE, ms=8, mec='white', mew=2)
    L = gt['field_side_m']
    ax.set_aspect('equal')
    ax.set_xlim(-0.15, L + 0.15)
    ax.set_ylim(-0.15, L + 0.15)
    ax.set_xlabel('x (m)', color=INK2)
    ax.set_ylabel('y (m)', color=INK2)
    ax.legend(loc='center', fontsize=8, frameon=False, labelcolor=INK2)
    ax.set_title(f'{run["name"]}: {run["shape"]} {run["side"]:.1f} m, {run["laps"]} laps, '
                 f'world frame (measured start pose)', color=INK, fontsize=10, loc='left')

    # zoom panels: one per tag, true-scale centimeters, SAME scale in every panel
    lim = 10.0
    for t, (m, C) in res['lms'].items():
        lim = max(lim, 1.1 * np.abs((ellipse(m, C) - np.array(gt['tag_xy'][t])[:, None]) * 100).max())
    lim = math.ceil(lim / 5) * 5
    for k, t in enumerate(sorted(res['lms'])):
        a = fig.add_subplot(gs[k // 4, 2 + k % 4])
        style(a)
        m, C = res['lms'][t]
        g = np.array(gt['tag_xy'][t])
        e = (ellipse(m, C) - g[:, None]) * 100
        a.plot(e[0], e[1], color=AQUA, lw=1.5)
        a.plot(*(100 * (m - g)), 'o', color=AQUA, ms=6, mec='white', mew=1)
        a.plot(0, 0, 'x', color=INK, ms=9, mew=2)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect('equal')
        err = 100 * np.linalg.norm(m - g)
        a.set_title(f'tag {t}: error {err:.1f} cm', color=INK, fontsize=9)
        a.set_xlabel('dx (cm)', color=INK2, fontsize=8)
        if k % 4 == 0:
            a.set_ylabel('dy (cm)', color=INK2, fontsize=8)
    fig.suptitle(f'Mean landmark error {100 * res["err"].mean():.1f} cm '
                 f'(shape only {100 * res["shape_err"].mean():.1f} cm); ellipses = 95% confidence '
                 f'(2.45 sigma); zoom panels centred on the true tag', color=INK2, fontsize=9,
                 x=0.01, ha='left', y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / f'map_{run["name"]}.png', dpi=160)
    plt.close(fig)


def dodge(labels, min_gap):
    """Spread end-of-line labels [(y, text, color, ax)] so none are closer than min_gap."""
    out = sorted(labels, key=lambda l: l[0])
    for i in range(1, len(out)):
        if out[i][0] - out[i - 1][0] < min_gap:
            out[i] = (out[i - 1][0] + min_gap,) + out[i][1:]
    return out


def fig_laps(runs, per_lap, plt):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax in (a1, a2):
        style(ax)
    labels = {a1: [], a2: []}
    last = max(r['lap'] for rs in per_lap.values() for r in rs)
    for run in runs:
        rows = per_lap[run['name']]
        laps = [r['lap'] for r in rows]
        c = SHAPE_COLOR[run['shape']]
        mk = 'o' if run['shape'] == 'square' else 's'
        e = [100 * r['res']['err'].mean() for r in rows]
        sh = [100 * r['res']['shape_err'].mean() for r in rows]
        sg = [100 * r['res']['sigma'].mean() for r in rows]
        a1.plot(laps, e, marker=mk, color=c, lw=2, ms=8, mec='white', mew=1.5)
        a1.plot(laps, sh, marker=mk, color=c, lw=1.2, ls=':', ms=6, mec='white', mew=1)
        labels[a1].append((e[-1], f'{run["shape"]} {e[-1]:.1f}', INK2))
        labels[a1].append((sh[-1], f'{run["shape"]} shape {sh[-1]:.1f}', MUTED))
        a2.plot(laps, sg, marker=mk, color=c, lw=2, ms=8, mec='white', mew=1.5)
        labels[a2].append((sg[-1], f'{run["shape"]} {sg[-1]:.1f}', INK2))
    for ax, items in labels.items():
        top = max(y for y, _, _ in items)
        placed = dodge(items, 0.07 * top)
        for y, text, color in placed:
            ax.text(last + 0.1, y, text, color=color, fontsize=8, va='center')
        data_top = max(line.get_ydata().max() for line in ax.get_lines() if len(line.get_ydata()))
        ax.set_ylim(0, 1.12 * max(data_top, max(y for y, _, _ in placed)))
    a1.set(xlabel='lap', ylabel='mean landmark error (cm)')
    a1.set_title('Landmark error after each lap (solid: world frame, dotted: shape only)',
                 color=INK, fontsize=9, loc='left')
    a2.set(xlabel='lap', ylabel='mean landmark std, sqrt(var_x+var_y) (cm)')
    a2.set_title('Landmark uncertainty after each lap', color=INK, fontsize=9, loc='left')
    for ax in (a1, a2):
        ax.set_xticks(sorted({r['lap'] for rs in per_lap.values() for r in rs}))
        ax.set_xlim(0.7, max(ax.get_xticks()) + 1.0)
        ax.xaxis.label.set_color(INK2)
        ax.yaxis.label.set_color(INK2)
    fig.tight_layout()
    fig.savefig(FIG / 'laps.png', dpi=160)
    plt.close(fig)


def fig_nis(runs, plt):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    style(ax)
    bins = np.linspace(0, 6, 25)
    for run in runs:
        ax.hist(run['nis'], bins=bins, density=True, histtype='step', lw=2,
                color=SHAPE_COLOR[run['shape']],
                label=f'{run["shape"]}: mean {run["nis"].mean():.2f} (n={run["nis"].size})')
    xs = np.linspace(0, 6, 200)
    ax.plot(xs, 0.5 * np.exp(-xs / 2), color=MUTED, lw=1.5, ls='--', label='chi-square, 2 DoF (mean 2)')
    ax.set_xlabel('NIS = y^T S^-1 y per landmark update', color=INK2)
    ax.set_ylabel('density', color=INK2)
    ax.set_title('Normalized innovation squared (consistency check of Q and R)', color=INK,
                 fontsize=9, loc='left')
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / 'nis.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='*', default=['square_final', 'octagon_final'])
    args = p.parse_args()

    gt = yaml.safe_load((ROOT / 'homework3/config/ground_truth.yaml').read_text())
    gt['tag_xy'] = {int(t): (float(v['x']), float(v['y'])) for t, v in gt['tags'].items()}
    runs = [load_run(n, gt) for n in args.runs]
    FIG.mkdir(parents=True, exist_ok=True)

    per_lap, csv_rows, md = {}, [], []
    md.append('# HW3 results (generated by homework3/evaluate.py)\n')
    md.append('World frame = measured start pose from config/ground_truth.yaml. '
              '"Shape" = mean error after the best rigid fit (start-pose error removed). '
              '"Scale" = best similarity-fit scale of the map vs truth (analysis only).\n')
    md.append('| run | lap | mapped | missed | mean err (cm) | max err (cm) | shape err (cm) | '
              'scale (%) | mean landmark std (cm) |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for run in runs:
        rows = []
        for snap in run['snaps']:
            res = evaluate_snapshot(snap, run, gt['tag_xy'])
            rows.append({'lap': snap['lap'], 'res': res})
            md.append(f'| {run["name"]} | {snap["lap"]} | {res["mapped"]} | '
                      f'{",".join(map(str, res["missed"])) or "-"} | {100 * res["err"].mean():.1f} | '
                      f'{100 * res["err"].max():.1f} | {100 * res["shape_err"].mean():.1f} | '
                      f'{res["scale_pct"]:+.1f} | {100 * res["sigma"].mean():.1f} |')
            for k, t in enumerate(res['ids']):
                csv_rows.append([run['name'], run['shape'], snap['lap'], t,
                                 f'{res["true"][k][0]:.3f}', f'{res["true"][k][1]:.3f}',
                                 f'{res["est"][k][0]:.4f}', f'{res["est"][k][1]:.4f}',
                                 f'{100 * res["err"][k]:.2f}', f'{100 * res["sigma"][k]:.2f}'])
        per_lap[run['name']] = rows
    md.append('\n## Path accuracy (from the logs, SLAM frame)\n')
    md.append('Estimated robot position at every stop (after the camera correction) vs the planned '
              'polygon. Independent of the start-pose measurement. "pose std" = the filter\'s own '
              'position uncertainty at those stops, i.e. how far the estimated path can be from the '
              'true one.\n')
    md.append('| run | lap | stops | cross-track mean (cm) | cross-track 95% (cm) | cross-track max (cm) '
              '| vertex error mean / max (cm) | pose std mean / max (cm) |')
    md.append('|---|---|---|---|---|---|---|---|')
    for run in runs:
        for lap, d in sorted(path_metrics(run).items()):
            x, s, v = 100 * d['xtrack'], 100 * d['std'], 100 * d.get('vertex', np.array([np.nan]))
            md.append(f'| {run["name"]} | {lap} | {x.size} | {x.mean():.1f} | {np.percentile(x, 95):.1f} | '
                      f'{x.max():.1f} | {np.nanmean(v):.1f} / {np.nanmax(v):.1f} | '
                      f'{s.mean():.1f} / {s.max():.1f} |')
    md.append('')
    for run in runs:
        res = per_lap[run['name']][-1]['res']
        md.append(f'## {run["name"]}: per-tag error after lap {per_lap[run["name"]][-1]["lap"]}\n')
        md.append('| tag | true (m) | estimated (m) | error (cm) | std (cm) |')
        md.append('|---|---|---|---|---|')
        for k, t in enumerate(res['ids']):
            md.append(f'| {t} | ({res["true"][k][0]:.3f}, {res["true"][k][1]:.3f}) | '
                      f'({res["est"][k][0]:.3f}, {res["est"][k][1]:.3f}) | '
                      f'{100 * res["err"][k]:.1f} | {100 * res["sigma"][k]:.1f} |')
        md.append(f'\nNIS: mean {run["nis"].mean():.2f} over {run["nis"].size} updates '
                  f'(2.0 expected if Q and R were exact).\n')
    (ROOT / 'homework3/output/results.md').write_text('\n'.join(md))
    with open(ROOT / 'homework3/output/results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['run', 'shape', 'lap', 'tag', 'true_x_m', 'true_y_m', 'est_x_m', 'est_y_m',
                    'error_cm', 'std_cm'])
        w.writerows(csv_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titleweight': 'normal'})
    fig_ground_truth(gt, runs, plt)
    for run in runs:
        fig_map(run, per_lap[run['name']][-1]['res'], gt, plt)
    fig_laps(runs, per_lap, plt)
    fig_nis(runs, plt)
    print('\n'.join(md))
    print(f'\nfigures -> {FIG}')


if __name__ == '__main__':
    main()
