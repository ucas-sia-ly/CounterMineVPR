"""Build a small deterministic GSV-Cities image manifest."""

import argparse
import json
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.mining.gsv_manifest import (  # noqa: E402
    DEFAULT_CITIES,
    build_gsv_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/GSVCities"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    # nargs="+" 表示可以指定多个城市，每个城市之间用空格分隔
    # 默认值为 DEFAULT_CITIES，即所有城市
    parser.add_argument("--places-per-city", type=int, default=1000)
    parser.add_argument("--images-per-place", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("cache/gsv_mini"))
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="omit selected places with missing images and continue",
    )
    # --allow-missing 是一个 boolean flag，用于指定是否允许在构建清单时忽略缺失的图像
    # 如果为 True，则在构建清单时会跳过缺失的图像，继续构建清单
    # 如果为 False，则在构建清单时会抛出异常，提示缺失的图像

    args = parser.parse_args()

    cities = list(dict.fromkeys(args.cities))
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest = build_gsv_manifest(
        dataset_root,
        cities=cities,
        places_per_city=args.places_per_city,
        images_per_place=args.images_per_place,
        seed=args.seed,
        allow_missing=args.allow_missing,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    summary_path = args.output_dir / "manifest_summary.json"
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "selected_cities": cities,
        "number_of_places": int(manifest["place_uid"].nunique()),
        "number_of_images": len(manifest),
        "images_per_place": args.images_per_place,
        "seed": args.seed,
        "number_of_missing_images": manifest.attrs["missing_image_count"],
        "dataset_root": str(dataset_root),
        "creation_command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "configuration": {
            "cities": cities,
            "places_per_city": args.places_per_city,
            "images_per_place": args.images_per_place,
            "seed": args.seed,
            "allow_missing": args.allow_missing,
            "output_dir": str(args.output_dir),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} images from {summary['number_of_places']} places")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
