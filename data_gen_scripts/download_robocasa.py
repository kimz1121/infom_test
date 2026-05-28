"""Standalone downloader for Robocasa target/atomic LeRobot tars.

Pulls Box links from the upstream robocasa repo's box_links_ds.json,
filters by task and split, downloads each tar, and extracts in place.

No robocasa package install required.
"""
import argparse
import json
import os
import os.path as osp
import shutil
import sys
import tarfile
import urllib.request
from urllib.error import URLError

BOX_LINKS_URL = (
    "https://raw.githubusercontent.com/robocasa/robocasa/main/"
    "robocasa/models/assets/box_links/box_links_ds.json"
)

ATOMIC_SEEN_18 = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]


def _to_direct(shared_url: str, ext: str = "tar") -> str:
    shared_id = shared_url.rstrip("/").split("/")[-1]
    base = shared_url.split("/s/")[0]
    return f"{base}/shared/static/{shared_id}.{ext}"


def _human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


def _download(url: str, dest_path: str):
    """Stream download with periodic percent updates (one line per 10%)."""
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "robocasa-infom-downloader"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out:
        total = resp.length
        read = 0
        next_mark = 10  # next percent to log
        chunk = 1 << 20  # 1 MiB
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            read += len(buf)
            if total:
                pct = 100 * read / total
                if pct >= next_mark:
                    print(f"  {_human(read)} / {_human(total)} ({pct:5.1f}%)")
                    next_mark = int(pct // 10 + 1) * 10
        print(f"  done {_human(read)}")


def _load_manifest(cache_path: str) -> dict:
    if osp.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    print(f"Fetching box_links_ds.json -> {cache_path}")
    os.makedirs(osp.dirname(cache_path), exist_ok=True)
    urllib.request.urlretrieve(BOX_LINKS_URL, cache_path)
    with open(cache_path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="~/.robocasa/raw",
                   help="Where to extract lerobot datasets")
    p.add_argument("--split", default="target", choices=["target", "pretrain"])
    p.add_argument("--category", default="atomic", choices=["atomic", "composite"],
                   help="atomic = single-skill demos; composite = multi-skill chained demos.")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Task names; default = 18 atomic-seen (only valid when --category atomic)")
    p.add_argument("--source", default="human", choices=["human", "mg"])
    p.add_argument("--probe_only", action="store_true",
                   help="Just print matching tar entries; no download")
    p.add_argument("--first_only", action="store_true",
                   help="Download only the first matching tar (size probe)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = osp.expanduser(args.out_dir)
    if args.category == "composite":
        if not args.tasks:
            sys.exit("--category composite requires --tasks <names...>")
        tasks = args.tasks
    else:
        tasks = args.tasks or ATOMIC_SEEN_18

    manifest = _load_manifest(osp.expanduser("~/.robocasa/box_links_ds.json"))

    # Pick the correct keys. Human: "{split}/{category}/{task}/{date}/lerobot.tar"
    # mg: "{split}/{category}/{task}/{date}/mg/demo/.../lerobot.tar"
    selected = []
    for task in tasks:
        prefix = f"{args.split}/{args.category}/{task}/"
        candidates = [k for k in manifest if k.startswith(prefix) and k.endswith("lerobot.tar")]
        if args.source == "human":
            candidates = [k for k in candidates if "/mg/" not in k]
        else:
            candidates = [k for k in candidates if "/mg/" in k]
        if not candidates:
            print(f"[skip] no match for {task} ({args.split}/{args.source})")
            continue
        candidates.sort()  # by date prefix
        selected.append((task, candidates[-1], manifest[candidates[-1]]))

    print(f"Matched {len(selected)} task tars under split={args.split} source={args.source}")
    for task, key, url in selected:
        print(f"  {task:35s} <- {key}")
    if args.probe_only:
        return

    if args.first_only and selected:
        selected = selected[:1]

    os.makedirs(out_dir, exist_ok=True)

    for task, key, shared_url in selected:
        # On-disk layout mirrors box key structure for easy lookup downstream.
        rel = key  # e.g. target/atomic/OpenCabinet/20250813/lerobot.tar
        target_tar = osp.join(out_dir, rel)
        target_extract = osp.dirname(target_tar)  # extract beside the tar

        marker = osp.join(target_extract, ".extracted")
        if osp.exists(marker) and not args.overwrite:
            print(f"[skip] already extracted: {target_extract}")
            continue

        os.makedirs(target_extract, exist_ok=True)
        direct = _to_direct(shared_url, ext="tar")
        try:
            _download(direct, target_tar)
        except URLError as e:
            print(f"[error] download failed for {task}: {e}")
            continue

        print(f"  extracting -> {target_extract}")
        with tarfile.open(target_tar, "r") as tf:
            tf.extractall(target_extract)
        os.remove(target_tar)
        with open(marker, "w") as f:
            f.write("ok\n")
        # Report size
        sz = 0
        for root, _, files in os.walk(target_extract):
            for fn in files:
                sz += osp.getsize(osp.join(root, fn))
        print(f"  done {task}: {_human(sz)} on disk")

    print("All requested tars processed.")


if __name__ == "__main__":
    main()
