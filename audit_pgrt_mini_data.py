"""Audit PGRT mini manifests after applying the configured data root."""

import argparse
import json
import os
import posixpath

import yaml


MANIFEST_KEYS = (
    "train_clean_json",
    "train_noisy_json",
    "valid_clean_json",
    "valid_noisy_json",
)


def load_rebased_manifest(path, data_root):
    with open(path, "r") as handle:
        entries = json.load(handle)
    if not data_root:
        return entries
    join_path = posixpath.join if data_root.startswith("/") else os.path.join
    return [
        join_path(
            data_root,
            os.path.basename(os.path.dirname(entry)),
            os.path.basename(entry),
        )
        for entry in entries
    ]


def assert_paired(clean_paths, noisy_paths, label):
    if len(clean_paths) != len(noisy_paths):
        raise AssertionError(label + " clean/noisy lengths differ")
    for clean_path, noisy_path in zip(clean_paths, noisy_paths):
        if os.path.basename(clean_path) != os.path.basename(noisy_path):
            raise AssertionError(label + " filename pairing mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="recipes/Mamba-SEUNet/PGRT-MPDM-v1-mini.yaml",
    )
    parser.add_argument("--skip-existence", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    data_config = config["data_cfg"]
    data_root = data_config.get("data_root")
    repository_root = os.path.dirname(os.path.abspath(__file__))
    manifests = {
        key: load_rebased_manifest(
            data_config[key]
            if os.path.isabs(data_config[key])
            else os.path.join(repository_root, data_config[key]),
            data_root,
        )
        for key in MANIFEST_KEYS
    }

    train_clean = manifests["train_clean_json"]
    train_noisy = manifests["train_noisy_json"]
    valid_clean = manifests["valid_clean_json"]
    valid_noisy = manifests["valid_noisy_json"]
    assert_paired(train_clean, train_noisy, "train")
    assert_paired(valid_clean, valid_noisy, "validation")

    train_names = {os.path.basename(path) for path in train_clean}
    valid_names = {os.path.basename(path) for path in valid_clean}
    overlap = sorted(train_names.intersection(valid_names))
    if overlap:
        raise AssertionError("train/validation overlap: " + overlap[0])

    missing = []
    if not args.skip_existence:
        missing = [
            path
            for entries in manifests.values()
            for path in entries
            if not os.path.isfile(path)
        ]
        if missing:
            raise FileNotFoundError(
                "{} manifest paths are missing; first: {}".format(
                    len(missing), missing[0]
                )
            )

    report = {
        "data_root": data_root,
        "train_pairs": len(train_clean),
        "validation_pairs": len(valid_clean),
        "train_validation_overlap": len(overlap),
        "missing_paths": len(missing),
    }
    print(json.dumps(report, indent=2))
    print("PASS PGRT mini manifest audit")


if __name__ == "__main__":
    main()
