import argparse
from pathlib import Path

import pandas as pd


DESCRIPTION = {
    "ADL1": "standing",
    "ADL2": "walking",
    "ADL3": "running",
    "ADL4": "jumping",
    "ADL5": "walking",  # upstairs
    "ADL6": "walking",  # downstairs
    # 7 and 8 are non detectable (only the transition) -> standing and sitting are the same upright posture
    "ADL7": "standing_to_standing",  # standing to sitting
    "ADL8": "standing_to_standing",  # sitting to standing
    "ADL9": "walking",  # uphill
    "ADL10": "walking",  # downhill
    "ADL11": "running",  # uphill
    "ADL12": "running",  # downhill
    "ADL13": "walking",  # stair-hopping
    "MO1": "walking-shooting",
    "MO2": "walking-shooting",  # quick engaging
    # 3-8 can only detect the transition and the "shooting" aspect.
    "MO3": "standing_to_standing-shooting",  # standing to kneeling shooting
    "MO4": "walking_to_standing-shooting",  # walking to kneeling shooting
    "MO5": "running_to_standing-shooting",  # running to kneeling shooting
    "MO6": "standing_to_down-shooting",  # intentional movement to prone
    "MO7": "walking_to_down-shooting",  # intentional movement to prone
    "MO8": "running_to_down-shooting",  # intentional movement to prone
    "MO9": "crawling",
    "FALL1": "standing_fall_down",  # frontal fall ending supine
    "FALL2": "standing_fall_down",  # frontal fall ending prone
    "FALL3": "standing_fall_down",  # back fall ending supine
    "FALL4": "standing_fall_down",  # left ending sideways
    "FALL5": "standing_fall_down",  # right ending sideways
}

TRIAL_COLUMNS = [
    "subject_id", "activity", "with_rifle", "repetition",
    "start_index", "end_index", "classes",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--hz", type=int, choices=[50, 100])
    parser.add_argument("--trials", type=Path)
    args = parser.parse_args()

    if args.trials:
        trials_path = args.trials
    elif args.root and args.hz:
        trials_path = args.root / "parquet" / f"{args.hz}hz" / "trials.csv"
    else:
        parser.error("Provide --trials or both --root and --hz")

    if trials_path.exists():
        trials = pd.read_csv(trials_path)
        trials["classes"] = trials["activity"].map(DESCRIPTION)
        trials.to_csv(trials_path, index=False)

        print(f"Updated {len(trials)} trials: {trials_path}")
        print(f"Trials with no class: {trials['classes'].isna().sum()}")
        return

    if not (args.root and args.hz):
        parser.error("Creating new trials requires --root and --hz")

    input_dir = args.root / "merged" / f"{args.hz}hz"
    output_dir = trials_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = []

    for i in range(1, 16):
        subject = f"ID{i}"
        files = sorted((input_dir / subject).glob("*.csv"))

        if not files:
            print(f"{subject}: no files found")
            continue

        data = []
        offset = 0

        for file in files:
            df = pd.read_csv(file).drop(
                columns=["timestamp_s"], errors="ignore"
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
                "end_index": end,
                "classes": DESCRIPTION[label],
            })

            data.append(df)
            offset = end

        pd.concat(data, ignore_index=True).to_parquet(
            output_dir / f"{subject}.parquet",
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        print(f"{subject}: {len(files)} trials, {offset} samples")

    result = pd.DataFrame(trials, columns=TRIAL_COLUMNS)
    result.to_csv(trials_path, index=False)

    print(f"Created {len(result)} trials: {trials_path}")
    print(f"Trials with no class: {result['classes'].isna().sum()}")


if __name__ == "__main__":
    main()