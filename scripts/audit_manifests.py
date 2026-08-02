import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_NAMES = {
    'train_clean': 'mini_train_clean_list.json',
    'train_noisy': 'mini_train_noisy_list.json',
    'val_clean': 'mini_val_clean_list.json',
    'val_noisy': 'mini_val_noisy_list.json',
}


def load_manifest(path):
    with path.open(encoding='utf-8') as manifest_file:
        entries = json.load(manifest_file)
    if not isinstance(entries, list) or not all(
        isinstance(entry, str) for entry in entries
    ):
        raise ValueError(f'{path} must contain a JSON list of paths.')
    return entries


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as manifest_file:
        for chunk in iter(lambda: manifest_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def basenames(entries):
    return [Path(entry).name for entry in entries]


def main():
    parser = argparse.ArgumentParser(
        description='Audit paired mini VoiceBank manifests without modifying them.'
    )
    parser.add_argument('--manifest-dir', default='data')
    parser.add_argument('--check-files', action='store_true')
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    paths = {
        name: manifest_dir / filename
        for name, filename in MANIFEST_NAMES.items()
    }
    manifests = {name: load_manifest(path) for name, path in paths.items()}
    errors = []

    for split in ('train', 'val'):
        clean = manifests[f'{split}_clean']
        noisy = manifests[f'{split}_noisy']
        if len(clean) != len(noisy):
            errors.append(
                f'{split} clean/noisy counts differ: {len(clean)} vs {len(noisy)}'
            )
        if basenames(clean) != basenames(noisy):
            errors.append(f'{split} clean/noisy basenames are not pairwise aligned')

    for name, entries in manifests.items():
        if len(entries) != len(set(entries)):
            errors.append(f'{name} contains duplicate paths')
        if args.check_files:
            missing = [entry for entry in entries if not Path(entry).is_file()]
            if missing:
                errors.append(
                    f'{name} has {len(missing)} missing files; first={missing[0]}'
                )

    train_names = set(basenames(manifests['train_clean']))
    val_names = set(basenames(manifests['val_clean']))
    overlap = sorted(train_names.intersection(val_names))
    if overlap:
        errors.append(
            f'train/validation basename overlap: {len(overlap)}; first={overlap[0]}'
        )

    for name, path in paths.items():
        entries = manifests[name]
        first = entries[0] if entries else '<empty>'
        print(
            f'{name}: count={len(entries)}, sha256={sha256(path)}, first={first}'
        )

    if errors:
        for error in errors:
            print(f'ERROR: {error}')
        raise SystemExit(1)
    print('Manifest audit passed: paired, unique, disjoint, and readable as requested.')


if __name__ == '__main__':
    main()
