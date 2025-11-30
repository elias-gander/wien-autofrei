import os
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import mapillary.interface as mly
from constants import (
    DEGREES_PER_METER_LON_VIENNA,
    DEGREES_PER_METER_LAT_VIENNA,
    MAPILLARY_ACCESS_TOKEN,
    MAPILLARY_CREATOR_ID,
)
import json
from PIL import Image
import requests
from io import BytesIO
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
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
mly.set_access_token(MAPILLARY_ACCESS_TOKEN)
for _, row in tqdm(points.iterrows(), total=points.shape[0]):
    id = row["objectid"]
    if id in processed_point_ids:
        continue

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
    cubemap_faces = [np.array(face) for face in cubemap_faces]
    # todo segmentation und inpainting auf cubemap faces
    sam = sam_model_registry["default"](
        checkpoint="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    )
    predictor = SamPredictor(sam)
    predictor.set_image(cubemap_faces[0])
    masks, _, _ = predictor.predict("car")
    # todo wie die bilder den cubemap faces zuordnen?
    cubemap_faces = {}
    pano = py360convert.c2e(cubemap_faces, 2048, 4096, cube_format="dict")
    # todo auf cloudflare laden
    Image.fromarray(pano.astype(np.uint8)).save("panorama.jpg", quality=75)
    with open("processed.txt", "a") as file:
        file.write(id)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
