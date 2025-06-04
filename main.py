from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io
import math

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
    list_of_centroids = kmeans.cluster_centers_.tolist()
    
    confusion_points = {
        "prota": [0.678, 0.501],
        "deutera": [-1.217, 0.782],
        "trita": [0.257, 0.0],
    }
    
    
    def get_r_theta(point_type: str):
        result_list = []
        for centroids in list_of_centroids:
            u_c = centroids[0]
            v_c = centroids[1]
            u_conf = confusion_points[point_type][0]
            v_conf = confusion_points[point_type][1]
            r = math.sqrt(((u_c - u_conf) ** 2) + ((v_c - v_conf) ** 2))
            
            seta = math.atan((v_c - v_conf) / (u_c - u_conf))
            
            result_list.append({
                "r": r,
                "seta": seta
            })
        return result_list
    
    list_of_r_prota = get_r_theta("prota")
    list_of_r_deutera = get_r_theta("deutera")
    list_of_r_trita = get_r_theta("trita")
    
    # print("list_of_r_prota: ", list_of_r_prota, end="\n")
    # print("list_of_r_deutera: ", list_of_r_deutera, end="\n")
    # print("list_of_r_trita: ", list_of_r_trita, end="\n")

    result_list = []

    for i in range(len(list_of_centroids)):
        result_list.append({
            "centroids": list_of_centroids[i],
            "r_setas": {
                "prota": list_of_r_prota[i],
                "deutera": list_of_r_deutera[i],
                "trita": list_of_r_trita[i]
            }
        })

    """
    [
        [0.654615, 0.6853421],
        [0.654615, 0.6853421],
        [0.654615, 0.6853421],
        [0.654615, 0.6853421],
    ]
    """

    """
    [
        {
            "centroids": [0.65654, 0.6468], [i]
            "r_setas": {
    #         "prota": list_of_r_prota, [i]
    #         "deutera": list_of_r_deutera, [i]
    #         "trita": list_of_r_trita [i]
    #     }
        }
    ]
    """
    # return {
    #     "centroids": list_of_centroids,
    #     "r_setas": {
    #         "prota": list_of_r_prota,
    #         "deutera": list_of_r_deutera,
    #         "trita": list_of_r_trita
    #     }
    # }

    print(result_list)

    return {"centroids": result_list}
