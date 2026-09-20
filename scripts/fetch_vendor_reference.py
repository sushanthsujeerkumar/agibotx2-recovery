"""Read selected members from the official public archive with verified HTTP ranges."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import requests


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        response = self.session.head(url, timeout=30)
        response.raise_for_status()
        self.size = int(response.headers['Content-Length'])
        self.position = 0
        self.bytes_read = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size-self.position)
        if size == 0:
            return b''
        if size > 8_000_000 or self.bytes_read + size > 12_000_000:
            raise ValueError('Bounded inspection exceeded byte budget')
        begin, end = self.position, self.position + size - 1
        response = self.session.get(self.url, headers={'Range': f'bytes={begin}-{end}'}, timeout=30)
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {begin}-{end}/{self.size}' or len(response.content) != size:
            raise ValueError('Server did not return the exact requested range')
        self.position += size
        self.bytes_read += size
        return response.content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Local cache outside the submission repository')
    args = parser.parse_args()
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    url = 'https://x2-aimdk.agibot.com/downloads/mc-x86-v1.0.0-20260522.zip'
    remote = RemoteZip(url)
    prefix = 'mc-x86-v1.0.0-20260522/bin/mc_param/src/robot/lx2501_3_t2d5/'
    selected = ['rl_models/policy_b_lie_up.onnx', 'rl/ground_up_config.yaml', 'rl/ground_init_config.yaml']
    manifest = {'source': url, 'archive_bytes': remote.size, 'members': {}}
    with zipfile.ZipFile(remote) as archive:
        (output/'archive_entries.json').write_text(json.dumps([
            {'name': member.filename, 'size': member.file_size, 'compressed_size': member.compress_size}
            for member in archive.infolist()], indent=2)+'\n')
        for name in selected:
            member = archive.getinfo(prefix+name)
            if member.file_size > 8_000_000:
                raise ValueError('Member exceeds bound')
            payload = archive.read(member)
            destination = output / Path(name).name
            if name.endswith('.onnx') and hashlib.sha256(payload).hexdigest() != '3faf3df8f9616448f9ae580e22c2fa28fcb25c1eb0c9a338a0b0c5b9a520522f':
                raise ValueError('Official download differs from inspected policy revision')
            destination.write_bytes(payload)
            manifest['members'][name] = {'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            print(name, manifest['members'][name], flush=True)
    manifest['transferred_bytes'] = remote.bytes_read
    (output/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
