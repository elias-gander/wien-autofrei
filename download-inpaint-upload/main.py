import os
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import mapillary.interface as mly
from constants import (
    BRUSHNET_OUTPUT_FOLDER_NAME,
    CUBEMAP_FACES_FOLDER_NAME,
    DEGREES_PER_METER_LON_VIENNA,
    DEGREES_PER_METER_LAT_VIENNA,
    LAMA_OUTPUT_FOLDER_NAME,
    LARGE_MASKS_FOLDER_NAME,
    MAPILLARY_ACCESS_TOKEN,
    MAPILLARY_CREATOR_ID,
    SMALL_MASKS_FOLDER_NAME,
)
import json
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from utils import classify_masks, merge_masks
import subprocess
import py360convert

processed_point_ids = set()
file_path = "processed_points.txt"

if not os.path.exists(file_path):
    open(file_path, "w", encoding="utf-8").close()

with open(file_path, "r+", encoding="utf-8") as file:
    for line in file:
        processed_point_ids.add(line.strip())

    file.seek(0, os.SEEK_END)
    if file.tell() > 0:
        file.seek(file.tell() - 1)
        last_char = file.read(1)
        if last_char != "\n":
            file.write("\n")

# points = gpd.read_file("sampled-points.gpkg", layer="kappazunder_image_punkte")
points = gpd.GeoDataFrame(
    {"objectid": ["123"]},
    geometry=[Point(16.379078, 48.198142)],
    crs="EPSG:4326",
)
# checkpoint from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
sam.to("mps")
sam_predictor = SamPredictor(sam)
mly.set_access_token(MAPILLARY_ACCESS_TOKEN)
for _, row in tqdm(points.iterrows(), total=points.shape[0]):
    id = row["objectid"]
    if id in processed_point_ids:
        continue

    # 1. download
    bbox = {
        "west": row.geometry.x - DEGREES_PER_METER_LON_VIENNA * 2,
        "south": row.geometry.y - DEGREES_PER_METER_LAT_VIENNA * 2,
        "east": row.geometry.x + DEGREES_PER_METER_LON_VIENNA * 2,
        "north": row.geometry.y + DEGREES_PER_METER_LAT_VIENNA * 2,
    }
    images_at_point = json.loads(mly.images_in_bbox(bbox))["features"]
    cubemap_face_ids = [
        image["properties"]["id"]
        for image in images_at_point
        if image["properties"]["creator_id"] == MAPILLARY_CREATOR_ID
    ]
    if len(cubemap_face_ids) != 4:
        raise RuntimeError(
            f"{len(cubemap_face_ids)} images found on mapillary for point {id}"
        )

    cubemap_face_urls = [
        mly.image_thumbnail(id, resolution=2048) for id in cubemap_face_ids
    ]
    cubemap_faces = [
        Image.open(BytesIO(requests.get(url).content)) for url in cubemap_face_urls
    ]

    # 2. segment
    for d in (
        CUBEMAP_FACES_FOLDER_NAME,
        SMALL_MASKS_FOLDER_NAME,
        LARGE_MASKS_FOLDER_NAME,
    ):
        subprocess.run(["rm", "-rf", d])
        os.makedirs(d)

    for i, face in enumerate(tqdm(cubemap_faces)):
        sam_predictor.set_image(np.array(face))
        masks, _, _ = sam_predictor.predict("car")
        masks = [m.astype(np.uint8) for m in masks]
        small_masks, large_masks = classify_masks(masks, threshold=0.1, dilate_px=15)
        small_masks = merge_masks(small_masks)
        large_masks = merge_masks(large_masks)
        face.save(os.path.join(CUBEMAP_FACES_FOLDER_NAME, f"face{i}.png"))
        Image.fromarray((small_masks * 255).astype(np.uint8)).save(
            f"{SMALL_MASKS_FOLDER_NAME}/face{i}.png"
        )
        Image.fromarray((large_masks * 255).astype(np.uint8)).save(
            f"{LARGE_MASKS_FOLDER_NAME}/face{i}.png"
        )

    # 3. inpaint
    should_run_lama = len(os.listdir(SMALL_MASKS_FOLDER_NAME)) > 0
    if should_run_lama:
        subprocess.run(
            [
                "iopaint",
                "run",
                "--model=lama",
                "--device=mps",
                "--image",
                CUBEMAP_FACES_FOLDER_NAME,
                "--mask",
                SMALL_MASKS_FOLDER_NAME,
                "--output",
                LAMA_OUTPUT_FOLDER_NAME,
            ]
        )
    should_run_brushnet = len(os.listdir(LARGE_MASKS_FOLDER_NAME)) > 0
    if should_run_brushnet:
        subprocess.run(
            [
                "iopaint",
                "run",
                "--model=brushnet",
                "--device=mps",
                "--image",
                CUBEMAP_FACES_FOLDER_NAME,
                "--mask",
                LARGE_MASKS_FOLDER_NAME,
                "--output",
                BRUSHNET_OUTPUT_FOLDER_NAME,
            ]
        )

    # 4. project
    # todo wie die bilder den cubemap faces zuordnen?
    inpainted_cubemap_faces = [
        Image.open(fname)
        for fname in os.listdir(
            BRUSHNET_OUTPUT_FOLDER_NAME
            if should_run_brushnet
            else LAMA_OUTPUT_FOLDER_NAME
        )
    ]
    inpainted_cubemap_faces = {}
    pano = py360convert.c2e(inpainted_cubemap_faces, 2048, 4096, cube_format="dict")

    # 5. upload
    Image.fromarray(pano.astype(np.uint8)).save("panorama.jpg", quality=75)

    with open("processed.txt", "a") as file:
        file.write(id)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
