#!/usr/bin/env python3
"""Reamostra e junta CHEST, LEFT e RIGHT por trial, sem arquivos de auditoria."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEVICES = ("CHEST", "LEFT", "RIGHT")
AXES = (("ax", "ay", "az"), ("wx", "wy", "wz"))
PATTERN = re.compile(r"^(ID\d+)_(.+)_r(\d+)_(CHEST|LEFT|RIGHT)$", re.I)


def read_streams(path):
    df = pd.read_csv(path)
    df["t"] = (df["timestamp_relative_s"] * 1000).round().astype("int64")
    streams = {}

    for axes in AXES:
        part = df[["t", *axes]].dropna().sort_values("t").drop_duplicates("t")
        streams[axes] = (part["t"].to_numpy(), part[list(axes)].to_numpy())

    return streams


def merge(streams, hz):
    start = max(t[0] for t, _ in streams.values())
    end = min(t[-1] for t, _ in streams.values())
    step = 1000 // hz
    grid = np.arange(((start + step - 1) // step) * step, end + 1, step)

    data = {"timestamp_s": grid / 1000}
    for device in DEVICES:
        for axes in AXES:
            t, values = streams[device, axes]
            for i, axis in enumerate(axes):
                data[f"{device.lower()}_{axis}"] = np.interp(grid, t, values[:, i])

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hz", type=int, nargs="+", choices=(50, 100), default=[50, 100])
    args = parser.parse_args()

    trials = {}
    for path in args.root.rglob("*.csv"):
        match = PATTERN.fullmatch(path.stem)
        if match:
            trial = f"{match[1].upper()}_{match[2].upper().replace('_', '')}_r{match[3]}"
            trials.setdefault(trial, {})[match[4].upper()] = path

    successes = 0
    errors = 0
    print(f"{len(trials)} trials encontrados")

    for i, (trial, paths) in enumerate(sorted(trials.items()), start=1):
        try:
            streams = {
                (device, axes): values
                for device in DEVICES
                for axes, values in read_streams(paths[device]).items()
            }

            for hz in args.hz:
                destination = args.output / f"{hz}hz" / trial.split("_")[0] / f"{trial}.csv"
                destination.parent.mkdir(parents=True, exist_ok=True)
                merge(streams, hz).to_csv(destination, index=False)
                successes += 1

        except Exception as exc:
            errors += 1
            print(f"ERRO {trial}: {exc}")

        if i % 25 == 0 or i == len(trials):
            print(f"{i}/{len(trials)} trials | {successes} arquivos | {errors} erros")


if __name__ == "__main__":
    main()
