"""Download the latest Copernicus lake/river water-level snapshot per station.

Credentials are read from CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY. Catalogue
CSVs are semicolon-delimited and contain many historical snapshots of the same
station; only the row with the newest content_date_end is downloaded.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


S3_PREFIX = "s3://eodata/"
S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
REQUIRED_COLUMNS = {
    "name",
    "content_length",
    "content_date_end",
    "s3_path",
}


def latest_station_rows(catalogue: Path) -> pd.DataFrame:
    data = pd.read_csv(catalogue, sep=";")
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Catalogue is missing columns: {sorted(missing)}")

    data["station"] = data["name"].str.extract(r"_([0-9]{13})_ALTI_")[0]
    data["end"] = pd.to_datetime(data["content_date_end"], errors="coerce")
    data["content_length"] = pd.to_numeric(data["content_length"], errors="coerce")
    data = data.dropna(subset=["station", "end", "content_length", "s3_path"])

    return (
        data.sort_values(["station", "end"])
        .drop_duplicates("station", keep="last")
        .reset_index(drop=True)
    )


def get_s3_client():
    access_key = os.environ.get("CDSE_S3_ACCESS_KEY")
    secret_key = os.environ.get("CDSE_S3_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(
            "Set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY before downloading."
        )

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install boto3 first: python -m pip install boto3") from exc

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="default",
    )


def download_one(client, row, output_dir: Path) -> tuple[str, str]:
    station = str(row.station)
    expected_size = int(row.content_length)
    destination = output_dir / f"{station}.geojson"

    if destination.exists() and destination.stat().st_size == expected_size:
        return station, "skipped"

    s3_path = str(row.s3_path)
    if not s3_path.startswith(S3_PREFIX):
        raise ValueError(f"Unexpected S3 path for station {station}: {s3_path}")

    key = s3_path.removeprefix(S3_PREFIX)
    temporary = destination.with_suffix(".geojson.part")
    temporary.unlink(missing_ok=True)

    try:
        client.download_file("eodata", key, str(temporary))
        actual_size = temporary.stat().st_size
        if actual_size != expected_size:
            raise IOError(
                f"Size mismatch for {station}: {actual_size} != {expected_size}"
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return station, "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download only the latest CDSE water-level file per station."
    )
    parser.add_argument("catalogue", type=Path, help="Downloaded lake or river CSV")
    parser.add_argument("output", type=Path, help="Destination directory")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect selection and size without authenticating or downloading",
    )
    args = parser.parse_args()

    latest = latest_station_rows(args.catalogue)
    total_gb = latest["content_length"].sum() / 1e9
    print(
        f"{args.catalogue.name}: {len(latest):,} latest stations, "
        f"{total_gb:.2f} GB"
    )
    print(
        f"records: {latest['end'].min()} through {latest['end'].max()}"
    )
    if args.dry_run:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    client = get_s3_client()
    failures: list[tuple[str, str]] = []
    counts = {"downloaded": 0, "skipped": 0}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(download_one, client, row, args.output): str(row.station)
            for row in latest.itertuples(index=False)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            station = futures[future]
            try:
                _, status = future.result()
                counts[status] += 1
            except Exception as exc:
                failures.append((station, str(exc)))
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"{completed:,}/{len(futures):,} processed | "
                    f"{counts['downloaded']:,} downloaded | "
                    f"{counts['skipped']:,} already present | "
                    f"{len(failures):,} failed"
                )

    if failures:
        print("\nFailed stations:")
        for station, error in failures[:20]:
            print(f"  {station}: {error}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    print(f"Complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
