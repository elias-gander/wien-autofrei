import os
import geopandas as gpd
from tqdm import tqdm
import mapillary.interface as mly
from constants import (
    DEGREES_PER_METER_LON_VIENNA,
    DEGREES_PER_METER_LAT_VIENNA,
    MAPILLARY_ACCESS_TOKEN,
)
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import py360convert


processed_point_ids = set()
with open("processed_points.txt", "a+") as file:
    file.seek(0)
    for line in file:
        processed_point_ids.add(line.strip())

    file.seek(-1, os.SEEK_END)
    if file.read(1) != b"\n":
        file.write("\n")

points = gpd.read_file("../sampled-points.gpkg", layer="kappazunder_image_punkte")
mly.set_access_token(MAPILLARY_ACCESS_TOKEN)
for index, point in tqdm(points.iterrows(), total=points.shape[0]):
    if processed_point_ids.has(index):
        continue

    bbox = {
        "west": point.geometry.x - DEGREES_PER_METER_LON_VIENNA / 2,
        "south": point.geometry.y - DEGREES_PER_METER_LAT_VIENNA / 2,
        "east": point.geometry.x + DEGREES_PER_METER_LON_VIENNA / 2,
        "north": point.geometry.y + DEGREES_PER_METER_LAT_VIENNA / 2,
    }
    images_at_point = mly.images_in_bbox(bbox)
    cubemap_face_ids = [
        image["id"]
        for image in images_at_point
        if image["creator"]["username"] == "eliasgander"
    ]
    if len(cubemap_face_ids) != 4:
        raise RuntimeError(
            f"{len(cubemap_face_ids)} images found on mapillary for point {index}"
        )

    cubemap_face_urls = [
        mly.image_thumbnail(id, resolution=2048) for id in cubemap_face_ids
    ]
    cubemap_faces = [
        Image.open(BytesIO(requests.get(url)) for url in cubemap_face_urls)
    ]
    # todo segmentation und inpainting auf cubemap faces
    # todo wie die bilder den cubemap faces zuordnen?
    cubemap_faces = {}
    cubemap_faces = {key: np.array(value) for key, value in cubemap_faces}
    pano = py360convert.c2e(cubemap_faces, 2048, 4096, cube_format="dict")
    # todo auf cloudflare laden
    Image.fromarray(pano.astype(np.uint8)).save("panorama.jpg", quality=75)
    with open("processed.txt", "a") as file:
        file.write(index)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
