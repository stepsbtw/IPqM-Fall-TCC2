"""Gera um PNG de 6 paineis por CSV ja mesclado (CHEST, LEFT e RIGHT).

Exemplo:
    python figuras_merged.py --root "C:/Projetos/TCC2/IPqM-Fall/merged" --output "C:/Projetos/TCC2/IPqM-Fall/figuras"

Se a raiz contiver as pastas 50hz/ e 100hz/, elas sao preservadas na saida.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEVICES = (
    ("chest", "Peito", "#2463A6"),
    ("left", "Punho esquerdo", "#B05A16"),
    ("right", "Punho direito", "#167568"),
)
SENSORS = (("ax", "ay", "az"), ("wx", "wy", "wz"))


def plot_trial(csv_path, destination, dpi):
    df = pd.read_csv(csv_path)
    time = df["timestamp_s"].to_numpy()

    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True, sharey="col")
    fig.suptitle(csv_path.stem.replace("_", "  ·  "), fontsize=20, fontweight="bold")
    axes[0, 0].set_title("Aceleração · magnitude (m/s²)", fontsize=15)
    axes[0, 1].set_title("Giroscópio · magnitude (rad/s)", fontsize=15)

    for row, (device, name, color) in enumerate(DEVICES):
        for col, axes_names in enumerate(SENSORS):
            columns = [f"{device}_{axis}" for axis in axes_names]
            magnitude = np.linalg.norm(df[columns].to_numpy(), axis=1)
            ax = axes[row, col]
            ax.plot(time, magnitude, color=color, linewidth=1)
            ax.grid(alpha=0.25)
            ax.set_ylim(bottom=0)
            if col == 0:
                ax.set_ylabel(name, fontsize=12, fontweight="bold")

    for ax in axes[-1]:
        ax.set_xlabel("Tempo relativo (s)")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--participants", type=int, nargs="+")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    files = sorted(args.root.rglob("ID*_r*.csv"))
    if args.participants:
        prefixes = tuple(f"ID{p}_" for p in args.participants)
        files = [path for path in files if path.name.startswith(prefixes)]

    print(f"{len(files)} trials encontrados")
    saved = 0
    for i, path in enumerate(files, 1):
        destination = args.output / path.relative_to(args.root).with_suffix(".png")
        try:
            plot_trial(path, destination, args.dpi)
            saved += 1
        except Exception as error:
            print(f"ERRO {path.name}: {error}")
        if i % 25 == 0 or i == len(files):
            print(f"{i}/{len(files)} trials | {saved} figuras")


if __name__ == "__main__":
    main()
