from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # หรือระบุเฉพาะ 'http://localhost:3000' ก็ได้
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

M_MATRIX = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.072175],
    [0.0193339, 0.119192, 0.9503041]
])

@app.post("/uv-space")
async def uv_space(image: UploadFile = File(...), k: int = Form(...)):
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]

    array_of_xyz = [np.dot(M_MATRIX, [r, g, b]) for r, g, b in pixels]

    array_of_luv = []
    for x, y, z in array_of_xyz:
        denom = x + 15 * y + 3 * z
        if denom == 0:
            continue
        u = (4 * x) / denom
        v = (9 * y) / denom
        array_of_luv.append([u, v])

    kmeans = KMeans(n_clusters=k, n_init=10).fit(array_of_luv)
    return {"centroids": kmeans.cluster_centers_.tolist()}
