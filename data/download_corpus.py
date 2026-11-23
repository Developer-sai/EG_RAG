"""Download the public policy document (Table 1 of the report) and record their SHA-256.

    python data/download_corpus.py

The file is saved as data/raw/D3.pdf. data/raw/manifest.csv records the URL, the
download date and the SHA-256 hash. If a file is already there it is not downloaded again;
its hash is checked against the manifest so a silently changed document is noticed.
If a site blocks scripted downloads, save the PDF from the browser as data/raw/<id>.pdf and
run this script again to record its hash.
"""
import csv
import datetime as dt
import hashlib
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RAW.mkdir(exist_ok=True)
    manifest_path = RAW / "manifest.csv"
    old = {}
    if manifest_path.exists():
        old = {r["id"]: r for r in csv.DictReader(manifest_path.open(encoding="utf-8"))}
    rows = []
    for src in csv.DictReader((HERE / "sources.csv").open(encoding="utf-8")):
        path = RAW / f"{src['id']}.pdf"
        if not path.exists():
            try:
                req = urllib.request.Request(src["url"], headers={"User-Agent": "Mozilla/5.0 (research download)"})
                path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
                print(f"downloaded {path.name}")
            except Exception as err:
                print(f"FAILED {src['id']}: {err}  -> save it manually as {path}")
                continue
        digest = sha256(path)
        prev = old.get(src["id"])
        if prev and prev["sha256"] != digest:
            print(f"WARNING {src['id']}: file differs from the recorded hash (document changed?)")
        date = prev["downloaded"] if prev and prev["sha256"] == digest else dt.date.today().isoformat()
        rows.append({"id": src["id"], "file": path.name, "url": src["url"], "downloaded": date, "sha256": digest})
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "file", "url", "downloaded", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}/1 public document present; manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
