"""Deterministic, image-level GSV-Cities manifest construction."""

import hashlib
from pathlib import Path
import random
import sys
from typing import Mapping, Sequence

import pandas as pd

# 默认城市
DEFAULT_CITIES = ("London", "Boston")

# 必须在数据集中存在的列
REQUIRED_COLUMNS = (
    "place_id",# 地点 ID，用于唯一标识每个地点
    "city_id",
    "year",
    "month",
    "northdeg",
    "lat",
    "lon",
    "panoid",
)

# 这些是构建 GSV-Cities 清单所需的列
MANIFEST_COLUMNS = (
    "row_index",# 行索引，用于唯一标识每个图像
    "image_id",
    "relative_path",# 图像的相对路径，用于加载图像
    "place_uid",# 唯一的地点标识符，由城市和地点 ID 组成
    "original_place_id",# 原始地点 ID，用于唯一标识每个地点
    "city_id",# 城市 ID，用于唯一标识每个城市
    "year",
    "month",
    "northdeg",# 北纬度，用于表示图像拍摄位置的纬度
    "lat",# 纬度，用于表示图像拍摄位置的纬度
    "lon",# 经度，用于表示图像拍摄位置的经度
    "panoid",# 全景 ID，用于唯一标识每个全景图像
)

# 构建 GSV-Cities 清单所需的图像文件名
def reconstruct_filename(row: Mapping[str, object]) -> str:
    """Match GSVCitiesDataset.get_img_name using the original place ID."""
    return (
        f"{row['city_id']}_{str(row['place_id']).zfill(7)}_"
        f"{str(row['year']).zfill(4)}_{str(row['month']).zfill(2)}_"
        f"{str(row['northdeg']).zfill(3)}_{row['lat']}_{row['lon']}_"
        f"{row['panoid']}.jpg"
    )

# 构建 GSV-Cities 清单所需的随机种子
def _stable_seed(seed: int, kind: str, identifier: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{kind}\0{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")

# 读取 GSV-Cities 数据集中的城市数据
def _read_city(dataset_root: Path, city: str) -> pd.DataFrame:
    csv_path = dataset_root / "Dataframes" / f"{city}.csv"
    dataframe = pd.read_csv(csv_path)
    absent = set(REQUIRED_COLUMNS).difference(dataframe.columns)
    if absent:
        raise ValueError(f"{csv_path} lacks required columns: {', '.join(sorted(absent))}")
    if dataframe.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(f"{csv_path} contains missing required metadata")
    if not pd.api.types.is_integer_dtype(dataframe["place_id"]):
        raise ValueError(f"{csv_path} has non-integer place_id values")
    if (dataframe["place_id"] < 0).any():
        raise ValueError(f"{csv_path} has negative place_id values")
    if not dataframe["city_id"].eq(city).all():
        raise ValueError(f"{csv_path} has city_id values other than {city}")

    dataframe = dataframe.loc[:, REQUIRED_COLUMNS].copy()
    dataframe["relative_path"] = dataframe.apply(
        lambda row: f"Images/{row['city_id']}/{reconstruct_filename(row)}", axis=1
    )
    return dataframe.drop_duplicates(subset="relative_path")

# 构建 GSV-Cities 清单
def build_gsv_manifest(
    dataset_root: str | Path,
    cities: Sequence[str] = DEFAULT_CITIES,
    places_per_city: int = 1000,
    images_per_place: int = 4,
    seed: int = 42,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Sample metadata deterministically and return rows in descriptor order.

    Each city is read independently, so peak metadata memory is bounded by one
    city CSV. If allow_missing is true, incomplete selected places are removed
    to keep exactly ``images_per_place`` images for every retained place.
    ``manifest.attrs['missing_image_count']`` records selected missing files.
    """
    if places_per_city <= 0 or images_per_place <= 0:
        raise ValueError("places_per_city and images_per_place must be positive")
    selected_cities = tuple(dict.fromkeys(cities))
    if not selected_cities:
        raise ValueError("cities must contain at least one city")

    root = Path(dataset_root).expanduser().resolve()
    records: list[dict[str, object]] = []
    for city in sorted(selected_cities):
        dataframe = _read_city(root, city)
        counts = dataframe.groupby("place_id", sort=True).size()
        eligible_ids = counts[counts >= images_per_place].index.tolist()
        place_rng = random.Random(_stable_seed(seed, "places", city))
        chosen_ids = set(
            place_rng.sample(eligible_ids, k=min(places_per_city, len(eligible_ids)))
        )

        selected = dataframe[dataframe["place_id"].isin(chosen_ids)]
        selected = selected.sort_values(["place_id", "relative_path"], kind="mergesort")
        for place_id, group in selected.groupby("place_id", sort=True):
            place_uid = f"{city}:{str(place_id).zfill(7)}"
            image_rng = random.Random(_stable_seed(seed, "images", place_uid))
            for row in image_rng.sample(group.to_dict("records"), k=images_per_place):
                relative_path = row["relative_path"]
                records.append(
                    {
                        "image_id": relative_path,
                        "relative_path": relative_path,
                        "place_uid": place_uid,
                        "original_place_id": place_id,
                        "city_id": row["city_id"],
                        "year": row["year"],
                        "month": row["month"],
                        "northdeg": row["northdeg"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "panoid": row["panoid"],
                    }
                )
        del dataframe, selected

    manifest = pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS[1:])
    manifest = manifest.sort_values(
        ["city_id", "place_uid", "relative_path"], kind="mergesort"
    ).reset_index(drop=True)

    missing = [
        path for path in manifest["relative_path"] if not (root / path).is_file()
    ]
    if missing:
        print(
            f"Missing {len(missing)} of {len(manifest)} selected GSV-Cities images under {root}",
            file=sys.stderr,
        )
        for path in missing[:5]:
            print(f"  {path}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more", file=sys.stderr)
        if not allow_missing:
            raise FileNotFoundError(f"{len(missing)} selected GSV-Cities images are missing")
        incomplete_places = set(
            manifest.loc[manifest["relative_path"].isin(missing), "place_uid"]
        )
        manifest = manifest[~manifest["place_uid"].isin(incomplete_places)].copy()
        manifest = manifest.reset_index(drop=True)
        print(
            f"Omitted {len(incomplete_places)} incomplete selected places",
            file=sys.stderr,
        )

    manifest.insert(0, "row_index", range(len(manifest)))
    manifest.attrs["missing_image_count"] = len(missing)
    return manifest.loc[:, MANIFEST_COLUMNS]
