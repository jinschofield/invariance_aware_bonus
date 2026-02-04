import argparse
import os
from typing import List, Tuple, Optional

import imageio.v2 as imageio
import numpy as np
import torch
import matplotlib.pyplot as plt

from ti.online.intrinsic import EpisodicEllipticalBonus
from ti.metrics.elliptical import elliptical_bonus


def build_state_onehots(grid_size: int, device: torch.device) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    positions = []
    onehots = []
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            vec = torch.zeros((grid_size * grid_size,), device=device)
            vec[idx] = 1.0
            positions.append((r, c))
            onehots.append(vec)
    return torch.stack(onehots, dim=0), positions


def step_pos(pos: Tuple[int, int], action: int, grid_size: int) -> Tuple[int, int]:
    r, c = pos
    if action == 0:  # up
        r -= 1
    elif action == 1:  # down
        r += 1
    elif action == 2:  # left
        c -= 1
    else:  # right
        c += 1
    r = int(np.clip(r, 0, grid_size - 1))
    c = int(np.clip(c, 0, grid_size - 1))
    return r, c


def compute_heatmap(
    Ainv: torch.Tensor,
    state_onehots: torch.Tensor,
    n_actions: int,
    beta: float,
    avg_actions: bool,
    grid_size: int,
) -> torch.Tensor:
    if avg_actions:
        actions = torch.arange(n_actions, device=state_onehots.device)
        z_rep = state_onehots.repeat_interleave(n_actions, dim=0)
        a_rep = actions.repeat(state_onehots.shape[0])
        a_onehot = torch.nn.functional.one_hot(a_rep.long(), num_classes=n_actions).float()
        phi = torch.cat([z_rep, a_onehot], dim=1)
        b = elliptical_bonus(phi, Ainv, beta=float(beta))
        b = b.view(state_onehots.shape[0], n_actions).mean(dim=1)
    else:
        a_rep = torch.zeros((state_onehots.shape[0],), device=state_onehots.device, dtype=torch.long)
        a_onehot = torch.nn.functional.one_hot(a_rep, num_classes=n_actions).float()
        phi = torch.cat([state_onehots, a_onehot], dim=1)
        b = elliptical_bonus(phi, Ainv, beta=float(beta))
    return b.view(grid_size, grid_size)


def render_heatmap(
    heat: np.ndarray,
    pos: Tuple[int, int],
    title: str,
    vmin: float,
    vmax: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(heat, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.scatter([pos[1]], [pos[0]], c="red", s=80, marker="x")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_state_timeseries(
    heatmaps: List[np.ndarray],
    pos_trace: List[Tuple[int, int]],
    grid_size: int,
    out_path: str,
) -> None:
    t_steps = np.arange(len(heatmaps))
    heat_arr = np.stack(heatmaps, axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in range(grid_size):
        for c in range(grid_size):
            y = heat_arr[:, r, c]
            label = f"({r},{c})"
            ax.plot(t_steps, y, label=label)

    for t, (r, c) in enumerate(pos_trace):
        ax.scatter([t], [heat_arr[t, r, c]], c="black", s=30, marker="o", zorder=3)

    ax.set_title("Elliptical bonus per state over time")
    ax.set_xlabel("t")
    ax.set_ylabel("bonus")
    ax.grid(alpha=0.3)
    ax.legend(title="state", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_visit_counts_timeseries(
    trace_rows: List[dict],
    grid_size: int,
    out_path: str,
    extrinsic_series: Optional[List[float]] = None,
) -> None:
    t_steps = np.arange(len(trace_rows))
    counts = {(r, c): np.zeros(len(trace_rows), dtype=float) for r in range(grid_size) for c in range(grid_size)}
    running = {(r, c): 0 for r in range(grid_size) for c in range(grid_size)}
    for i, row in enumerate(trace_rows):
        pos = (row["row"], row["col"])
        running[pos] += 1
        for key in running:
            counts[key][i] = running[key]

    fig, ax = plt.subplots(figsize=(6, 4))
    for r in range(grid_size):
        for c in range(grid_size):
            ax.plot(t_steps, counts[(r, c)], label=f"({r},{c})")
    if extrinsic_series is not None:
        hit_idx = np.nonzero(np.array(extrinsic_series) > 0.0)[0]
        if hit_idx.size > 0:
            ax.scatter(hit_idx, np.zeros_like(hit_idx), c="red", s=25, marker="x", label="extrinsic hit")
    ax.set_title("Cumulative visits per state over time")
    ax.set_xlabel("t")
    ax.set_ylabel("visit count")
    ax.grid(alpha=0.3)
    ax.legend(title="state", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_intrinsic_extrinsic_timeseries(
    intrinsic: List[float],
    extrinsic: List[float],
    out_path: str,
) -> None:
    t_steps = np.arange(len(intrinsic))
    intrinsic_arr = np.array(intrinsic)
    extrinsic_arr = np.array(extrinsic)
    total = intrinsic_arr + extrinsic_arr

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_steps, intrinsic_arr, label="intrinsic bonus")
    ax.plot(t_steps, extrinsic_arr, label="extrinsic reward")
    ax.plot(t_steps, total, label="total (intrinsic + extrinsic)")
    hit_idx = np.nonzero(extrinsic_arr > 0.0)[0]
    if hit_idx.size > 0:
        ax.scatter(hit_idx, total[hit_idx], c="red", s=30, marker="x", label="extrinsic hit")
    ax.set_title("Intrinsic bonus vs extrinsic hits")
    ax.set_xlabel("t")
    ax.set_ylabel("reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_state_trace(
    pos_trace: List[Tuple[int, int]],
    grid_size: int,
    extrinsic: List[float],
    out_path: str,
) -> None:
    t_steps = np.arange(len(pos_trace))
    idx = np.array([r * grid_size + c for r, c in pos_trace], dtype=float)
    extrinsic_arr = np.array(extrinsic)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_steps, idx, label="state index")
    hit_idx = np.nonzero(extrinsic_arr > 0.0)[0]
    if hit_idx.size > 0:
        ax.scatter(hit_idx, idx[hit_idx], c="red", s=30, marker="x", label="extrinsic hit")
    ax.set_title("State visitation trace (goal hits marked)")
    ax.set_xlabel("t")
    ax.set_ylabel("state index")
    ax.set_yticks(np.arange(grid_size * grid_size))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=2)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument(
        "--no-avg-actions",
        action="store_true",
        help="Disable action-averaging when computing heatmaps.",
    )
    parser.add_argument(
        "--print-values",
        action="store_true",
        help="Print raw heatmap values at each timestep.",
    )
    parser.add_argument("--out-dir", type=str, default="elliptical_bonus_demo/outputs")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument(
        "--extrinsic-demo",
        action="store_true",
        help="Add extrinsic reward on goal state and plot interactions.",
    )
    parser.add_argument("--goal-row", type=int, default=0)
    parser.add_argument("--goal-col", type=int, default=0)
    parser.add_argument("--extrinsic-reward", type=float, default=1.0)
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Save per-step visit count + bonus-at-pos trace to CSV.",
    )
    parser.add_argument(
        "--verify-visit-decay",
        action="store_true",
        help="Compute visit-count vs bonus correlation and summary stats.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    state_onehots, positions = build_state_onehots(args.grid_size, device)
    z_dim = state_onehots.shape[1]
    bonus = EpisodicEllipticalBonus(
        z_dim=z_dim,
        n_actions=args.n_actions,
        beta=args.beta,
        lam=args.lam,
        num_envs=1,
        device=device,
    )

    pos_idx = int(rng.integers(0, len(positions)))
    pos = positions[pos_idx]

    heatmaps = []
    pos_trace = []
    visit_counts = {(r, c): 0 for r in range(args.grid_size) for c in range(args.grid_size)}
    trace_rows = []
    avg_actions = not args.no_avg_actions
    intrinsic_series = []
    extrinsic_series = []
    goal = (int(args.goal_row), int(args.goal_col))
    for t in range(int(args.steps)):
        Ainv = bonus.Ainv[0]
        heat = compute_heatmap(
            Ainv,
            state_onehots,
            args.n_actions,
            args.beta,
            avg_actions,
            args.grid_size,
        )
        heatmaps.append(heat.detach().cpu().numpy())
        pos_trace.append(pos)
        bonus_at_pos = float(heat[pos[0], pos[1]].item())
        ext = float(args.extrinsic_reward) if args.extrinsic_demo and pos == goal else 0.0
        intrinsic_series.append(bonus_at_pos)
        extrinsic_series.append(ext)
        visit_before = visit_counts[pos]
        trace_rows.append(
            {
                "t": int(t),
                "row": int(pos[0]),
                "col": int(pos[1]),
                "visit_count": int(visit_before),
                "bonus_at_pos": bonus_at_pos,
                "extrinsic_reward": ext,
                "total_reward": bonus_at_pos + ext,
            }
        )
        if args.print_values:
            heat_np = heat.detach().cpu().numpy()
            print(f"\n[t={t}] bonus heatmap:")
            print(np.array2string(heat_np, precision=6, suppress_small=False))

        action = int(rng.integers(0, args.n_actions))
        state_idx = pos[0] * args.grid_size + pos[1]
        z = state_onehots[state_idx].unsqueeze(0)
        a = torch.tensor([action], device=device)
        bonus.compute_and_update(z, a)
        visit_counts[pos] += 1
        pos = step_pos(pos, action, args.grid_size)

    vmin = float(np.min(heatmaps))
    vmax = float(np.max(heatmaps))

    frame_paths = []
    for t, (heat, cur_pos) in enumerate(zip(heatmaps, pos_trace)):
        out_path = os.path.join(frames_dir, f"heatmap_{t:04d}.png")
        render_heatmap(
            heat,
            cur_pos,
            title=f"Elliptical bonus (t={t})",
            vmin=vmin,
            vmax=vmax,
            out_path=out_path,
        )
        frame_paths.append(out_path)

    render_state_timeseries(
        heatmaps,
        pos_trace,
        args.grid_size,
        out_path=os.path.join(args.out_dir, "state_bonus_timeseries.png"),
    )
    render_visit_counts_timeseries(
        trace_rows,
        args.grid_size,
        out_path=os.path.join(args.out_dir, "visit_counts_timeseries.png"),
        extrinsic_series=extrinsic_series if args.extrinsic_demo else None,
    )
    if args.extrinsic_demo:
        render_intrinsic_extrinsic_timeseries(
            intrinsic_series,
            extrinsic_series,
            out_path=os.path.join(args.out_dir, "intrinsic_extrinsic_timeseries.png"),
        )
        render_state_trace(
            pos_trace,
            args.grid_size,
            extrinsic_series,
            out_path=os.path.join(args.out_dir, "state_trace_with_extrinsic.png"),
        )

    images = [imageio.imread(p) for p in frame_paths]
    gif_path = os.path.join(args.out_dir, "bonus_evolution.gif")
    imageio.mimsave(gif_path, images, duration=1.0 / max(1, args.fps))
    print(f"Saved {len(frame_paths)} frames to {frames_dir}")
    print(f"Saved GIF to {gif_path}")

    if args.save_trace:
        trace_path = os.path.join(args.out_dir, "visit_bonus_trace.csv")
        with open(trace_path, "w", newline="", encoding="utf-8") as f:
            f.write("t,row,col,visit_count,bonus_at_pos,extrinsic_reward,total_reward\n")
            for row in trace_rows:
                f.write(
                    f"{row['t']},{row['row']},{row['col']},"
                    f"{row['visit_count']},{row['bonus_at_pos']},"
                    f"{row['extrinsic_reward']},{row['total_reward']}\n"
                )
        print(f"Saved visit/bonus trace to {trace_path}")

    if args.verify_visit_decay:
        visits = np.array([r["visit_count"] for r in trace_rows], dtype=float)
        bonuses = np.array([r["bonus_at_pos"] for r in trace_rows], dtype=float)
        if len(visits) > 1 and np.std(visits) > 0 and np.std(bonuses) > 0:
            corr = float(np.corrcoef(visits, bonuses)[0, 1])
        else:
            corr = float("nan")
        by_count = {}
        for r in trace_rows:
            by_count.setdefault(r["visit_count"], []).append(r["bonus_at_pos"])
        count_summary = {k: float(np.mean(v)) for k, v in sorted(by_count.items())}
        summary_path = os.path.join(args.out_dir, "visit_bonus_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"visit_count_vs_bonus_corr={corr}\n")
            f.write("mean_bonus_by_visit_count:\n")
            for k, v in count_summary.items():
                f.write(f"  {k}: {v}\n")
        print(f"Visit-count/bonus correlation: {corr}")
        print(f"Saved visit/bonus summary to {summary_path}")

        t_total = len(trace_rows)
        early_cut = max(1, t_total // 3)
        late_start = max(0, (2 * t_total) // 3)
        early_rows = [r for r in trace_rows if r["t"] < early_cut]
        late_rows = [r for r in trace_rows if r["t"] >= late_start]
        early_means = {}
        late_means = {}
        for r in early_rows:
            early_means.setdefault(r["visit_count"], []).append(r["bonus_at_pos"])
        for r in late_rows:
            late_means.setdefault(r["visit_count"], []).append(r["bonus_at_pos"])
        early_means = {k: float(np.mean(v)) for k, v in sorted(early_means.items())}
        late_means = {k: float(np.mean(v)) for k, v in sorted(late_means.items())}

        equiv_path = os.path.join(args.out_dir, "visit_bonus_equivalence.txt")
        with open(equiv_path, "w", encoding="utf-8") as f:
            f.write(f"early_window_t=[0,{early_cut - 1}]\n")
            f.write(f"late_window_t=[{late_start},{t_total - 1}]\n")
            f.write("early_mean_bonus_by_visit_count:\n")
            for k, v in early_means.items():
                f.write(f"  {k}: {v}\n")
            f.write("late_mean_bonus_by_visit_count:\n")
            for k, v in late_means.items():
                f.write(f"  {k}: {v}\n")
            if early_means and late_means:
                f.write("best_late_match_for_each_early_count:\n")
                for k, v in early_means.items():
                    best_k = min(late_means, key=lambda lk: abs(late_means[lk] - v))
                    diff = abs(late_means[best_k] - v)
                    f.write(f"  early {k} -> late {best_k} (diff={diff})\n")
            if 1 in early_means and 2 in late_means:
                diff = abs(early_means[1] - late_means[2])
                f.write(f"diff_early1_vs_late2={diff}\n")
        print(f"Saved visit/bonus equivalence to {equiv_path}")


if __name__ == "__main__":
    main()
