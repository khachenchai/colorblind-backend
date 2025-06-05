from fastapi import FastAPI, UploadFile, File, Form, Request
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

array_of_luv = []
list_of_pixel_clusters = []


@app.post("/uv-space")
async def uv_space(image: UploadFile = File(...), k: int = Form(...), confType: str = 'prota'):
    print('confType', confType)
    # เปิดรูปภาพและแปลงเป็น RGB
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    width, height = img.size
    pixels = [img.getpixel((x, y)) for y in range(height) for x in range(width)]

    pixel_data = []
    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            pixel_data.append({
                "r": r,
                "g": g,
                "b": b,
                "x": x,
                "y": y
            })

    # Normalize RGB
    normalized_pixels = [[r / 255, g / 255, b / 255] for r, g, b in pixels]

    # XYZ matrix
    array_of_xyz = [np.dot(M_MATRIX, [r, g, b]) for r, g, b in normalized_pixels]

    # UV space + เก็บค่า LUV
    # global array_of_luv
    # array_of_luv = []
    array_of_uv = []

    for idx, (x, y, z) in enumerate(array_of_xyz):
        denom = x + 15 * y + 3 * z
        if denom == 0:
            continue
        u = (4 * x) / denom
        v = (9 * y) / denom

        if y > 0.008856:
            l = 116 * (y ** (1 / 3)) - 16
        else:
            l = 903.3 * y

        confusion_points = {
            "prota": [0.678, 0.501],
            "deutera": [-1.217, 0.782],
            "trita": [0.257, 0.0],
        }
        u_conf, v_conf = confusion_points[confType]
        
        r = math.sqrt((u - u_conf)**2 + (v - v_conf)**2)

        array_of_uv.append([u, v])
        array_of_luv.append({
            "l": l,
            "u": u,
            "v": v,
            "r": r
        })

    # K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10).fit(array_of_uv)
    list_of_centroids = kmeans.cluster_centers_.tolist()
    
    for idx, cluster_num in enumerate(kmeans.labels_):
        pixel_info = pixel_data[idx]
        list_of_pixel_clusters.append({
            "pixel": {
                "r": pixel_info["r"],
                "g": pixel_info["g"],
                "b": pixel_info["b"]
            },
            "position": {
                "x": pixel_info["x"],
                "y": pixel_info["y"]
            },
            "cluster_num": int(cluster_num)
        })
        array_of_luv[idx]['position'] = {
            "x": pixel_info["x"],
            "y": pixel_info["y"]
        }

    confusion_points = {
        "prota": [0.678, 0.501],
        "deutera": [-1.217, 0.782],
        "trita": [0.257, 0.0],
    }

    def get_r_theta(point_type: str):
        result_list = []
        u_conf, v_conf = confusion_points[point_type]

        for u_c, v_c in list_of_centroids:
            r = math.sqrt((u_c - u_conf)**2 + (v_c - v_conf)**2)
            seta = math.atan2((v_c - v_conf), (u_c - u_conf))  # ใช้ atan2 แทน
            result_list.append({
                "r": r,
                "seta": seta
            })
        return result_list

    list_of_r_prota = get_r_theta("prota")
    list_of_r_deutera = get_r_theta("deutera")
    list_of_r_trita = get_r_theta("trita")

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

    return {
        "centroids": result_list,
        "array_of_luv": array_of_luv,  # ส่งกลับถ้าต้องใช้, ไม่งั้นตัดออก
        "array_of_pixels": list_of_pixel_clusters
    }





def remap_for_prota_and_deutera(centriods, cluster_type, curr_setas, sorted_m_k, seta_prime_new_after, seta_prime_new_previous):
    result_list = []

    for repeat in range(2):  # 2 รอบซ้ำ
        # 1. เรียง cluster ใหม่ตามค่า m_k
        # (ใช้ sorted_m_k อยู่แล้ว)

        for m in range(1, 21):
            new_setas = {}
            seta_v = 0
            # คงค่ามุมเดิม (fix มุมเดิมไปก่อน)
            # a = (seta_prime_new_after + seta_prime_new_previous) * 0.5
            if cluster_type == 'prota':
                seta_v = -1.4988
            elif cluster_type == 'deutera':
                seta_v = 1.6428
            
            a = (math.atan((math.tan(seta_prime_new_after - seta_v) - math.tan(seta_prime_new_previous - seta_v)))) + seta_v

            for c in sorted_m_k:
                cluster_num = c['cluster_num']
                curr_seta = curr_setas[cluster_num]

                b = curr_seta + (m / 100)

                if (a >= b or a <= b):
                    new_seta = curr_seta - (m / 100)
                else:
                    new_seta = a
                
                r_value = centriods[cluster_num - 1]['r_setas'][cluster_type]['r']

                # ✅ เก็บผลลัพธ์รอบนี้
                if repeat == 1:
                    result_list.append({
                        "cluster_num": cluster_num,
                        "m": m / 100,
                        "r": r_value,
                        "new_seta": new_seta
                    })

                # ✅ เก็บ new_seta ของ cluster นี้ สำหรับรอบถัดไป
                new_setas[cluster_num] = new_seta

            # ✅ หลังจบ cluster ทั้งหมดของรอบ m → ใช้ new_seta เป็น curr_seta รอบถัดไป
            curr_setas = new_setas.copy()

    return result_list
    # return

def remap_for_trita(centriods, curr_setas, sorted_m_k, seta_prime_new_after, seta_prime_new_previous):
    result_list = []

    for repeat in range(2):  # 2 รอบซ้ำ
        # 1. เรียง cluster ใหม่ตามค่า m_k
        # (ใช้ sorted_m_k อยู่แล้ว)

        for m in range(1, 21):
            new_setas = {}

            # คงค่ามุมเดิม (fix มุมเดิมไปก่อน)
            a = (seta_prime_new_after + seta_prime_new_previous) * 0.5

            for c in sorted_m_k:
                cluster_num = c['cluster_num']
                curr_seta = curr_setas[cluster_num]

                b = curr_seta + (m / 100)

                if (a >= b or a <= b):
                    new_seta = curr_seta - (m / 100)
                else:
                    new_seta = a
                
                r_value = centriods[cluster_num - 1]['r_setas']['trita']['r']

                # ✅ เก็บผลลัพธ์รอบนี้
                if repeat == 1:
                    result_list.append({
                        "cluster_num": cluster_num,
                        "m": m / 100,
                        "r": r_value,
                        "new_seta": new_seta
                    })

                # ✅ เก็บ new_seta ของ cluster นี้ สำหรับรอบถัดไป
                new_setas[cluster_num] = new_seta

            # ✅ หลังจบ cluster ทั้งหมดของรอบ m → ใช้ new_seta เป็น curr_seta รอบถัดไป
            curr_setas = new_setas.copy()

    return result_list

@app.post('/analysis')
async def analysis(req: Request):
    body = await req.json()
    r_seta_type = body.get("r_seta_type")
    centriods = body.get("centriods")

    list_of_seta = []
    list_of_m = []

    for i in range(len(centriods)):
        # print(centriods[i]['r_setas'][r_seta_type]['seta'])
        list_of_seta.append({
            "cluster_num": i + 1,
            "seta": centriods[i]['r_setas'][r_seta_type]['seta']
        })

    n = len(list_of_seta)
    for k in range(n):
        m_k = ((list_of_seta[k]['seta'] - list_of_seta[k - 1]['seta']) ** 2) + ((list_of_seta[(k + 1) % n]['seta'] - list_of_seta[k]['seta']) ** 2)
        # print(M)
        list_of_m.append({
            "cluster_num": list_of_seta[k]['cluster_num'],
            "cluster_seta": list_of_seta[k]['seta'],
            "m_k": m_k
        })
    
    sorted_m_k = sorted(list_of_m, key=lambda x: x['m_k'], reverse=True)
    # print(sorted_m_k)


    # ค่าเริ่มต้นที่ได้มา
    seta_prime_new_after = sorted_m_k[1]['cluster_seta']
    seta_prime_new_previous = sorted_m_k[-1]['cluster_seta']

    print("seta_prime_new_after:", seta_prime_new_after)
    print("seta_prime_new_previous:", seta_prime_new_previous)

    # สร้าง dict เก็บค่า curr_seta ของแต่ละ cluster
    curr_setas = {c['cluster_num']: c['cluster_seta'] for c in sorted_m_k}

    if (r_seta_type == 'remap_for_trita'):
        remap_result = remap_for_trita(centriods, curr_setas, sorted_m_k, seta_prime_new_after, seta_prime_new_previous)
    else:
        remap_result = remap_for_prota_and_deutera(centriods, r_seta_type, curr_setas, sorted_m_k, seta_prime_new_after, seta_prime_new_previous)

    # remap_for_trita(curr_setas, sorted_m_k, seta_prime_new_after, seta_prime_new_previous)
    remap_payload_result = {
        "type": r_seta_type,
        "result": remap_result
    }

    u_confs = {
        "prota": 0.678,
        "deutera": -1.217,
        "trita": 0.257
    }

    v_confs = {
        "prota": 0.501,
        "deutera": 0.782,
        "trita": 0.0
    }

    translate_u_v = {
        "type": r_seta_type,
        "result": []
    }

    # cluster_to_new_seta = {entry['cluster_num']: entry['new_seta'] for entry in remap_payload_result}
    cluster_to_new_seta = {entry['cluster_num']: entry['new_seta'] for entry in remap_payload_result['result']}
    cluster_to_new_r = []

    print(f"Total pixels: {len(list_of_pixel_clusters)}")
    print(f"Total LUV entries: {len(array_of_luv)}")
    print(f"Cluster to new_seta map: {cluster_to_new_seta}")

    # print('array_of_luv', array_of_luv)

    # สร้าง dict เพื่อ map cluster_num -> new_seta
    # luv_pixel = []
    # สำหรับทุก pixel
    for idx, pixel in enumerate(list_of_pixel_clusters):
        cluster_num = pixel['cluster_num']
        pixel_position = pixel['position']

        # หา new_seta
        new_seta = cluster_to_new_seta.get(cluster_num)

        # print(f"\nPixel idx: {idx}")
        # print(f"  Cluster num: {cluster_num}")
        # print(f"  Pixel position: {pixel_position}")
        # print(f"  new_seta found: {new_seta}")

        if new_seta is not None:
            # ไปหา pixel ใน array_of_luv ที่ position ตรงกัน
            luv_pixel = array_of_luv[idx]
            if luv_pixel['position']['x'] == pixel_position['x'] and luv_pixel['position']['y'] == pixel_position['y']:
                # เพิ่ม new_seta
                luv_pixel['new_seta'] = new_seta
                cluster_to_new_r.append(luv_pixel)
    
    print(cluster_to_new_r)

    # for i in range(len(list_of_pixel_clusters)):
    #     curr_pixel_data = list_of_pixel_clusters[i]
        # curr_cluster = 
        # new_seta = next(
        #     (entry["new_seta"] for entry in remap_result if entry["cluster_num"] == list_of_pixel_clusters),
        #     None
        # )

        # u_prime = u_confs[remap_payload_result["type"]] + (data['r'] * math.cos(remap_payload_result["result"]))
        # v_prime = v_confs[remap_payload_result["type"]] + (data['r'] * math.sin(data['new_seta']))
        
        # translate_u_v["result"].append({
        #     # "cluster_num": data['cluster_num'],
        #     # "m": data['m'],
        #     # "r": data['r'],
        #     # "new_seta": data['new_seta'],
        #     "u": u_prime,
        #     "v": v_prime
        # })

    # for i in range(len(remap_payload_result['result'])):
    #     data = remap_payload_result['result'][i]

    #     u_prime = u_confs[remap_payload_result["type"]] + (data['r'] * math.cos(data['new_seta']))
    #     v_prime = v_confs[remap_payload_result["type"]] + (data['r'] * math.sin(data['new_seta']))

    #     translate_u_v["result"].append({
    #         "cluster_num": data['cluster_num'],
    #         "m": data['m'],
    #         "r": data['r'],
    #         "new_seta": data['new_seta'],
    #         "u": u_prime,
    #         "v": v_prime
    #     })
    



    # print(centriods[0]['r_setas'])

    
    # print('array_of_luv:', array_of_luv)
    # print("r_seta_type:", r_seta_type)
    # print("centriods:", centriods)

    # print(translate_u_v)

    return {'msg': 'wow'}