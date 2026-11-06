r"""Research data lives in a GitHub release of this repo, not on the PC (disk space). This tool fetches it back
into work/ when a study needs it, and can free the local copies again.

  .venv\Scripts\python.exe -m tools.data_sync pull            # download + unpack into work/
  .venv\Scripts\python.exe -m tools.data_sync status          # what is local vs in the release
  .venv\Scripts\python.exe -m tools.data_sync free            # delete local copies that are safely in the release

The release (tag below) holds research-collected.tar.gz (Kalshi price histories, odds, weather, logs) and
research-derived.tar.gz (features/context/style tables, walk-forward predictions, replay caches).
"""
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "DiegoDeLaVara/tennis-edge"
TAG = "research-data-2026-11-06"
ASSETS = ["research-collected.tar.gz", "research-derived.tar.gz"]
WORK = Path(__file__).resolve().parents[1] / "work"


def url(name):
    return f"https://github.com/{REPO}/releases/download/{TAG}/{name}"


def members(name):
    data = urllib.request.urlopen(url(name)).read()
    return data, tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").getnames()


def pull():
    WORK.mkdir(exist_ok=True)
    for name in ASSETS:
        data = urllib.request.urlopen(url(name)).read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():                      # only plain files/dirs inside work/
                if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                    continue
                tf.extract(m, WORK)
        print(f"unpacked {name} ({len(data) / 1e6:.0f} MB)")


def status():
    for name in ASSETS:
        try:
            _, names = members(name)
        except Exception as e:  # noqa
            print(f"{name}: not reachable in release {TAG} ({e})")
            continue
        local = [n for n in names if (WORK / n).exists()]
        print(f"{name}: {len(names)} entries in the release, {len(local)} also on this PC")


def free():
    total = 0
    for name in ASSETS:
        _, names = members(name)                           # only delete what the release really holds
        for n in names:
            p = WORK / n
            if p.is_file():
                total += p.stat().st_size
                p.unlink()
    for d in [p for p in WORK.iterdir() if p.is_dir()]:
        if not any(d.iterdir()):
            d.rmdir()
    print(f"freed {total / 1e6:.0f} MB; run 'pull' to get the data back")


if __name__ == "__main__":
    {"pull": pull, "status": status, "free": free}.get(sys.argv[1] if len(sys.argv) > 1 else "status", status)()
