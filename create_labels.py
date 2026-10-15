import argparse
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile


TRIAL_CLASS = {
    "ADL1": "standing",
    "ADL2": "walking",
    "ADL3": "running",
    "ADL9": "walking",  # Uphill.
    "ADL10": "walking",  # Downhill.
    "ADL11": "running",  # Uphill.
    "ADL12": "running",  # Downhill.
    "ADL7": "standing_transition_sitting",
    "ADL8": "sitting_transition_standing",
    "MO3": "standing_transition_kneeling",  # Shooting stance.
    "MO4": "walking_transition_kneeling",  # Shooting stance.
    "MO5": "running_transition_kneeling",  # Shooting stance.
    "MO6": "standing_transition_prone",  # Shooting stance.
    "MO7": "walking_transition_prone",  # Shooting stance.
    "MO8": "running_transition_prone",  # Shooting stance.
    "MO9": "crawling",
    "FALL1": "standing_frontalfall_supine",
    "FALL2": "standing_frontalfall_prone",
    "FALL3": "standing_backfall_supine",
    "FALL4": "standing_sidefall_sideways",
    "FALL5": "standing_sidefall_sideways",
    "ADL4": "jumping",  # Pending: Jumping. repeated jumps every 2 seconds.
    "ADL5": "walking",  # Pending: Stair climb. (walking)
    "ADL6": "walking",  # Pending: Stair descent. (walking)
    "ADL13": "walking",  # Pending: Stair climb. HOPPING 2 degrees (walking).
    "MO1": "sweep",  # Pending: Military sweep (shooting stance) while walking. (walking) POINTING GUN
    "MO2": "sweep",  # Pending: Military sweep (shooting stance) while quick engaging. (walking) POINTING GUN
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    path = parser.parse_args().trials

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        trials = list(reader)

    if "classes" not in columns:
        columns.append("classes")

    for trial in trials:
        trial["classes"] = TRIAL_CLASS[trial["activity"]]

    temp = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="",
            dir=path.parent, delete=False,
        ) as f:
            temp = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(trials)

        temp.replace(path)
    finally:
        if temp:
            temp.unlink(missing_ok=True)

    print(f"Updated: {path}")
    print(f"Pending trial class: {sum(t['classes'] == '' for t in trials)}")


if __name__ == "__main__":
    main()