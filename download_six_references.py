"""Fetch the pinned published comparison tables. Never generate chess results."""
import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest() == entry['source_sha256']


def download(directory, catalog):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(catalog).read_text())['files']
    seen = set()
    for entry in entries:
        name = entry['name']
        if not re.fullmatch(r'[KQRBNP]+v[KQRBNP]+\.rtbw', name) or name in seen:
            raise ValueError('invalid or duplicate table name')
        if not re.fullmatch(r'[a-f0-9]{64}', entry['source_sha256']) or entry['bytes'] <= 0:
            raise ValueError('invalid source digest or size')
        seen.add(name)
    downloaded = 0
    for entry in entries:
        name = entry['name']
        path = root / name
        if verify(path, entry):
            continue
        pieces = len(name.split('.')[0]) - 1
        tier = '3-4-5-wdl' if pieces <= 5 else f'{pieces}-wdl'
        url = f'https://tablebase.lichess.ovh/tables/standard/{tier}/{name}'
        temporary = None
        try:
            with urllib.request.urlopen(url, timeout=120) as source, tempfile.NamedTemporaryFile(dir=root, delete=False) as output:
                temporary = Path(output.name)
                total = 0
                while block := source.read(1 << 20):
                    total += len(block)
                    if total > entry['bytes']:
                        raise ValueError(f'{name}: response exceeds pinned size')
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if not verify(temporary, entry):
                raise ValueError(f'{name}: source size/SHA-256 mismatch')
            os.replace(temporary, path)
            downloaded += 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dict(files=len(entries), downloaded=downloaded, all_sources_verified=True,
                bytes=sum(entry['bytes'] for entry in entries), generated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--catalog', default=str(Path(__file__).with_name('six_source_catalog.json')))
    args = parser.parse_args()
    print(json.dumps(download(args.directory, args.catalog), indent=2))
