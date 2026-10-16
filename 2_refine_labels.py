import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--hz", type=int, choices=[50, 100], required=True)
    args = parser.parse_args()

    trials = pd.read_csv(args.trials)
    half_window = round(0.5 * args.hz)
    segments = []

    for subject, subject_trials in trials.groupby("subject_id", sort=False):
        data = pd.read_parquet(args.parquet / f"{subject}.parquet")

        for _, trial in subject_trials.iterrows():
            start = int(trial["start_index"])
            end = int(trial["end_index"])
            labels = str(trial["labels"])

            if "fall" not in labels:
                segments.append(trial.to_dict())
                continue

            acceleration = data.iloc[start:end][
                ["chest_ax", "chest_ay", "chest_az"]
            ].to_numpy()

            magnitude = np.linalg.norm(acceleration, axis=1)
            peak = start + int(np.argmax(magnitude))

            fall_start = max(start, peak - half_window)
            fall_end = min(end, peak + half_window)

            labels = [
                ("standing", start, fall_start),
                ("fall", fall_start, fall_end),
                ("down", fall_end, end),
            ]

            for label, segment_start, segment_end in labels:
                if segment_start >= segment_end:
                    continue

                segment = trial.to_dict()
                segment["start_index"] = segment_start
                segment["end_index"] = segment_end
                segment["labels"] = label
                segments.append(segment)

        print(f"{subject}: processed {len(subject_trials)} trials")

    result = pd.DataFrame(segments, columns=trials.columns)

    output = args.trials.parent / "segments.csv"
    result.to_csv(output, index=False)

    print(f"Original trials: {len(trials)}")
    print(f"Output segments: {len(result)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()