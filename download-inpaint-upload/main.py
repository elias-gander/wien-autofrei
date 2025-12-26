import os
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import mapillary.interface as mly
from constants import (
    CUBEMAP_FACE_SIZE_PX,
    CUBEMAP_FACES_FOLDER_NAME,
    DEGREES_PER_METER_LON_VIENNA,
    DEGREES_PER_METER_LAT_VIENNA,
    DETECTION_VALUES_TO_INPAINT,
    INPAINTED_CUBEMAP_FACES_FOLDER_NAME,
    MAPILLARY_ACCESS_TOKEN,
    MAPILLARY_CREATOR_ID,
    MASKS_FOLDER_NAME,
)
from utils import recreate_folder
import json
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import base64
import cv2
import mapbox_vector_tile
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
    {"objectid": ["123"], "direction": [120]},
    geometry=[Point(16.379078, 48.198142)],
    crs="EPSG:4326",
)
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

    cubemap_face_metadata = {
        id: json.loads(mly.image_from_key(id))["features"]["properties"]
        for id in cubemap_face_ids
    }
    cubemap_faces = {
        id: Image.open(
            BytesIO(
                requests.get(face_metadata["thumb_2048_url"].replace("\\", "")).content
            )
        ).resize((CUBEMAP_FACE_SIZE_PX, CUBEMAP_FACE_SIZE_PX))
        for id, face_metadata in cubemap_face_metadata.items()
    }
    recreate_folder(CUBEMAP_FACES_FOLDER_NAME)
    for id, face in cubemap_faces.items():
        face.save(f"{CUBEMAP_FACES_FOLDER_NAME}/{id}.jpg")

    # 2. create masks
    recreate_folder(MASKS_FOLDER_NAME)
    for id, face in cubemap_faces.items():
        detections = mly.get_detections_with_image_id(id)
        decoded_detections = [
            mapbox_vector_tile.decode(
                base64.decodebytes(feature.properties.pixel_geometry.encode("utf-8"))
            )
            for feature in detections.features
            if feature.properties.value in DETECTION_VALUES_TO_INPAINT
        ]
        vehicle_mask_polygons = [
            [
                [
                    coord[0] / mask["mpy-or"]["extent"] * CUBEMAP_FACE_SIZE_PX,
                    CUBEMAP_FACE_SIZE_PX
                    - 1
                    - coord[1] / mask["mpy-or"]["extent"] * CUBEMAP_FACE_SIZE_PX,
                ]
                for coords in polygon["geometry"]["coordinates"]
                for coord in coords
            ]
            for mask in decoded_detections
            for polygon in mask["mpy-or"]["features"]
        ]
        mask = np.zeros((CUBEMAP_FACE_SIZE_PX, CUBEMAP_FACE_SIZE_PX), dtype=np.uint8)
        for polygon in vehicle_mask_polygons:
            coords = [(p[0], p[1]) for p in polygon]
            cv2.fillPoly(mask, [np.array(coords, dtype=np.int32)], 255)

        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            iterations=1,
        )
        Image.fromarray(mask).save(f"{MASKS_FOLDER_NAME}/{id}.png")

    # 3. inpaint
    subprocess.run(
        [
            "iopaint",
            "run",
            "--model=lama",
            "--device=mps",
            "--image",
            CUBEMAP_FACES_FOLDER_NAME,
            "--mask",
            MASKS_FOLDER_NAME,
            "--output",
            INPAINTED_CUBEMAP_FACES_FOLDER_NAME,
        ],
        check=True,
    )

    # 4. project
    def cubemap_image_id_with_compass_angle_closest_to(angle):
        return min(
            cubemap_face_ids,
            key=lambda id: abs(cubemap_face_metadata[id]["compass_angle"] - angle),
        )

    cubemap_face_direction_to_image_id_dict = {
        "F": cubemap_image_id_with_compass_angle_closest_to(row["direction"]),
        "R": cubemap_image_id_with_compass_angle_closest_to(
            (row["direction"] + 90) % 360
        ),
        "B": cubemap_image_id_with_compass_angle_closest_to(
            (row["direction"] + 180) % 360
        ),
        "L": cubemap_image_id_with_compass_angle_closest_to(
            (row["direction"] + 270) % 360
        ),
        "U": None,
        "D": None,
    }
    dummy_cubemap_face = Image.new(
        "RGB", (CUBEMAP_FACE_SIZE_PX, CUBEMAP_FACE_SIZE_PX), (128, 128, 128)
    )
    cubemap_face_direction_to_image_dict = {
        direction: (
            np.array(
                Image.open(f"{INPAINTED_CUBEMAP_FACES_FOLDER_NAME}/{id}.png")
                if id
                else dummy_cubemap_face
            )
        )
        for direction, id in cubemap_face_direction_to_image_id_dict.items()
    }
    pano = py360convert.c2e(
        cubemap_face_direction_to_image_dict,
        CUBEMAP_FACE_SIZE_PX,
        CUBEMAP_FACE_SIZE_PX * 2,
        cube_format="dict",
    )

    # 5. upload
    Image.fromarray(pano.astype(np.uint8)).save("panorama.jpg", quality=75)

    with open("processed.txt", "a") as file:
        file.write(id)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
