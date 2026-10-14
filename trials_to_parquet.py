
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--hz", type=int, choices=[50, 100], required=True)
args = parser.parse_args()

INPUT = args.root / "merged" / f"{args.hz}hz"
OUTPUT = args.root / "parquet" / f"{args.hz}hz"
OUTPUT.mkdir(parents=True, exist_ok=True)

trials = []

for i in range(1, 16):
    subject = f"ID{i}"
    files = sorted((INPUT / subject).glob("*.csv"))

    if not files:
        print(f"{subject}: no files found")
        continue

    data = []
    offset = 0

    for file in files:
        df = pd.read_csv(file)

        df = df.drop(
            columns=["timestamp_s"],
            errors="ignore"
        )

        name = file.stem.removeprefix(f"{subject}_")
        activity, repetition = name.rsplit("_", 1)
        repetition = int(repetition.removeprefix("r"))

        with_rifle = int(activity.endswith("R"))
        label = activity[:-1] if with_rifle else activity

        start = offset
        end = start + len(df)

        trials.append({
            "subject_id": subject,
            "activity": label,
            "with_rifle": with_rifle,
            "repetition": repetition,
            "start_index": start,
            "end_index": end
        })

        data.append(df)
        offset = end

    pd.concat(data, ignore_index=True).to_parquet(
        OUTPUT / f"{subject}.parquet",
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    print(f"{subject}: {len(files)} trials, {offset} samples")

pd.DataFrame(
    trials,
    columns=[
        "subject_id", "label", "with_rifle",
        "repetition", "start_index", "end_index"
    ]
).to_csv(OUTPUT / "trials.csv", index=False)

print(f"Saved {len(trials)} trials")
