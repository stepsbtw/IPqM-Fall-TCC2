"""Gera figuras PNG diretamente do trials.csv e dos Parquets por participante.

Saida organizada por atividade:
    output/
      ADL1/
        ID1_ADL1_r1.png
        ID2_ADL1_r1.png
      FALL1/
        ID1_FALL1_r1.png
        ...

Cada linha do trials.csv descreve um trial com intervalo [start_trial, end_trial),
em indices posicionais do respectivo IDx.parquet (end_trial EXCLUSIVO).
O parquet contem os 18 canais: chest_*, left_* e right_*.

Exemplo (Windows):
    python figures_parquet_by_activity_simple.py \
        --trials "C:/Projetos/TCC2/IPqM-Fall/dataset/parquet/50hz/trials.csv" \
        --parquet-dir "C:/Projetos/TCC2/IPqM-Fall/dataset/parquet/50hz" \
        --output "C:/Projetos/TCC2/IPqM-Fall/figures/50hz" \
        --hz 50 --only-falls --magnitude-only

Dependencias: pip install pandas numpy matplotlib pyarrow
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEVICES = (
    ("chest", "Chest"),
    ("left", "Left wrist"),
    ("right", "Right wrist"),
)
SENSORS = (("ax", "ay", "az"), ("wx", "wy", "wz"))
AXIS_COLORS = {"x": "#d4483b", "y": "#27955e", "z": "#3479c5"}
MAGNITUDE_COLOR = "#202020"
CHANNELS = [
    f"{device}_{axis}"
    for device, _ in DEVICES
    for axis in ("ax", "ay", "az", "wx", "wy", "wz")
]

ACTIVITY_INFO = {
    "ADL1":  "Standing",
    "ADL2":  "Walking",
    "ADL3":  "Running",
    "ADL4":  "Jumping",
    "ADL5":  "Stair climb",
    "ADL6":  "Stair descend",
    "ADL7":  "Standing -> Sitting",
    "ADL8":  "Sitting -> Standing",
    "ADL9":  "Uphill walk",
    "ADL10": "Downhill walk",
    "ADL11": "Uphill run",
    "ADL12": "Downhill run",
    "ADL13": "Stair hop",
    "MO1":   "Sweep Walking",
    "MO2":   "Sweep Engaging",
    "MO3":   "Standing -> Kneeling Shooting",
    "MO4":   "Walking -> Kneeling Shooting",
    "MO5":   "Running -> Kneeling Shooting",
    "MO6":   "Standing -> Prone Shooting",
    "MO7":   "Walking -> Prone Shooting",
    "MO8":   "Running -> Prone Shooting",
    "MO9":   "Crawl",
    "FALL1": "Standing -> Frontal Fall -> Supine",
    "FALL2": "Standing -> Frontal Fall -> Prone",
    "FALL3": "Standing -> Backward Fall -> Supine",
    "FALL4": "Standing -> Right Fall -> Right Sideways",
    "FALL5": "Standing -> Left Fall -> Left Sideways",
}



def plot_trial(data, trial_row, destination, hz, dpi, magnitude_only=False):
    start, end = int(trial_row.start_trial), int(trial_row.end_trial)
    trial = data.iloc[start:end]
    time = np.arange(len(trial)) / hz

    activity_code = str(trial_row.activity)
    activity_name = ACTIVITY_INFO.get(activity_code, activity_code)
    subject = str(trial_row.subject_id)
    repetition = int(trial_row.repetition)
    rifle = bool(int(trial_row.with_rifle))

    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True)

    rifle_text = "with rifle" if rifle else "without rifle"
    fig.suptitle(
        f"{activity_code} · {activity_name} · {subject} · repetition {repetition} · {rifle_text}",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    signal_label = "Magnitude" if magnitude_only else "X, Y, Z and magnitude"
    axes[0, 0].set_title(f"Accelerometer · {signal_label}", fontsize=13)
    axes[0, 1].set_title(f"Gyroscope · {signal_label}", fontsize=13)

    for row, (device, name) in enumerate(DEVICES):
        for col, component_names in enumerate(SENSORS):
            ax = axes[row, col]
            values = trial[[f"{device}_{a}" for a in component_names]].to_numpy(dtype=float)
            if not magnitude_only:
                for k, axis_name in enumerate("xyz"):
                    ax.plot(
                        time,
                        values[:, k],
                        color=AXIS_COLORS[axis_name],
                        label=axis_name.upper(),
                        linewidth=0.9,
                        alpha=0.9,
                    )
            ax.plot(
                time,
                np.linalg.norm(values, axis=1),
                color=MAGNITUDE_COLOR,
                label="Magnitude",
                linewidth=1.35,
            )
            ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=8, ncol=4)
            if col == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold")

            if "fall_start" in trial_row.index and "fall_end" in trial_row.index:
                a, b = trial_row["fall_start"], trial_row["fall_end"]
                if pd.notna(a) and pd.notna(b) and start <= a < b <= end:
                    t1, t2 = (int(a) - start) / hz, (int(b) - start) / hz
                    ax.axvspan(t1, t2, color="#efb44c", alpha=0.10)
                    ax.axvline(t1, color="#da891a", linestyle="--", linewidth=1.25)
                    ax.axvline(t2, color="#17874c", linestyle="--", linewidth=1.25)

    for ax in axes[-1]:
        ax.set_xlabel("Relative trial time (s)")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True, help="Path to trials.csv")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        required=True,
        help="Directory containing ID1.parquet, ID2.parquet, etc.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory for PNG files")
    parser.add_argument("--hz", type=float, default=50, help="Sampling rate (default: 50)")
    parser.add_argument("--participants", type=int, nargs="+", help="Example: 1 2 3")
    parser.add_argument("--activities", nargs="+", help="Example: FALL1 FALL2")
    parser.add_argument("--only-falls", action="store_true", help="Only activities FALL*")
    parser.add_argument("--magnitude-only", action="store_true", help="Plot only vector magnitude")
    parser.add_argument("--limit", type=int, help="Limit number of figures (for testing)")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    if args.hz <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--hz and --limit must be positive")

    trials = pd.read_csv(args.trials)
    required = {"subject_id", "activity", "with_rifle", "repetition", "start_trial", "end_trial"}
    missing = required - set(trials.columns)
    if missing:
        parser.error(f"Missing columns in trials.csv: {sorted(missing)}")

    if args.participants:
        wanted = {f"ID{i}" for i in args.participants}
        trials = trials[trials.subject_id.isin(wanted)]
    if args.activities:
        trials = trials[trials.activity.isin(args.activities)]
    if args.only_falls:
        trials = trials[trials.activity.str.startswith("FALL", na=False)]
    if args.limit:
        trials = trials.head(args.limit)

    print(f"{len(trials)} trials selected")
    saved = 0
    total = len(trials)

    for subject, group in trials.groupby("subject_id", sort=False):
        parquet_path = args.parquet_dir / f"{subject}.parquet"
        if not parquet_path.exists():
            print(f"WARNING: Parquet not found: {parquet_path}")
            continue
        try:
            data = pd.read_parquet(parquet_path, columns=CHANNELS)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("Install pyarrow: pip install pyarrow") from exc

        for trial_row in group.itertuples(index=False):
            start, end = int(trial_row.start_trial), int(trial_row.end_trial)
            if not 0 <= start < end <= len(data):
                print(f"WARNING: Invalid interval: {subject} {trial_row.activity} [{start}, {end})")
                continue

            suffix = "R" if int(trial_row.with_rifle) else ""
            activity_code = str(trial_row.activity)
            file_name = f"{subject}_{activity_code}{suffix}_r{int(trial_row.repetition)}.png"
            destination = args.output / activity_code / file_name
            plot_trial(data, pd.Series(trial_row._asdict()), destination, args.hz, args.dpi, args.magnitude_only)
            saved += 1
            if saved % 25 == 0 or saved == total:
                print(f"{saved}/{total} figures created")

        print(f"{subject}: {len(group)} trials processed")

    print(f"Done: {saved} PNG figures in {args.output}")


if __name__ == "__main__":
    main()
