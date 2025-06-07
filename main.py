from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io
import math
import statistics
import base64
# from colormath.color_diff import delta_e_cie2000
from skimage.color import deltaE_ciede2000
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor

import colorspacious

# Define the config for ΔE2000
cie_config = {
    "name": "CIEDE2000",
    "cvd": None  # Optional: simulate color vision deficiency
}

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
    array_of_xyz = []
    for r, g, b in normalized_pixels:
        rgb_vector = np.array([r, g, b]).reshape((3, 1))
        xyz_vector = np.dot(M_MATRIX, rgb_vector).flatten()
        array_of_xyz.append(xyz_vector)

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
        "array_of_pixels": list_of_pixel_clusters,
        "original_pixels": [p["pixel"] for p in list_of_pixel_clusters]
    }





def remap_for_prota_and_deutera(centriods, cluster_type, m_body_value, curr_setas, sorted_m_k, seta_prime_new_after, seta_prime_new_previous):
    result_list = []

    for repeat in range(2):  # 2 รอบซ้ำ
        # 1. เรียง cluster ใหม่ตามค่า m_k
        # (ใช้ sorted_m_k อยู่แล้ว)

        # for m in range(1, 21):
        new_setas = {}
        seta_v = 0
        # คงค่ามุมเดิม (fix มุมเดิมไปก่อน)
        # a = (seta_prime_new_after + seta_prime_new_previous) * 0.5
        if cluster_type == 'prota':
            seta_v = -1.4988
        elif cluster_type == 'deutera':
            seta_v = 1.6428
        
        x1 = math.cos(seta_prime_new_after - seta_v)
        y1 = math.sin(seta_prime_new_after - seta_v)
        x2 = math.cos(seta_prime_new_previous - seta_v)
        y2 = math.sin(seta_prime_new_previous - seta_v)

        avg_x = (x1 + x2) / 2
        avg_y = (y1 + y2) / 2
        
        # a = (math.atan2((math.tan(seta_prime_new_after - seta_v) - math.tan(seta_prime_new_previous - seta_v)))) + seta_v
        a = math.atan2(avg_y, avg_x) + seta_v

        for c in sorted_m_k:
            cluster_num = c['cluster_num']
            curr_seta = curr_setas[cluster_num]
            if curr_seta is None:
                print(f"❌ curr_seta not found for cluster_num: {cluster_num}")
                continue  # skip or handle this cluster
                
            if m_body_value is None:
                print("❌ ERROR: m_body_value is None")
                m_body_value = 0.01  # fallback default

            b = curr_seta + (m_body_value)

            if (a >= b or a <= b):
                new_seta = curr_seta - (m_body_value)
            else:
                new_seta = a
            
            r_value = centriods[cluster_num - 1]['r_setas'][cluster_type]['r']

            # ✅ เก็บผลลัพธ์รอบนี้
            if repeat == 1:
                result_list.append({
                    "cluster_num": cluster_num,
                    "m": m_body_value,
                    "r": r_value,
                    "new_seta": new_seta
                })

            # ✅ เก็บ new_seta ของ cluster นี้ สำหรับรอบถัดไป
            new_setas[cluster_num] = new_seta

        # ✅ หลังจบ cluster ทั้งหมดของรอบ m → ใช้ new_seta เป็น curr_seta รอบถัดไป
        curr_setas = new_setas.copy()

    return result_list
    # return

def remap_for_trita(centriods, curr_setas, m_body_value, sorted_m_k, seta_prime_new_after, seta_prime_new_previous):
    result_list = []

    for repeat in range(2):  # 2 รอบซ้ำ
        # 1. เรียง cluster ใหม่ตามค่า m_k
        # (ใช้ sorted_m_k อยู่แล้ว)

        # for m in range(1, 21):
        new_setas = {}

        # คงค่ามุมเดิม (fix มุมเดิมไปก่อน)
        a = (seta_prime_new_after + seta_prime_new_previous) * 0.5

        for c in sorted_m_k:
            cluster_num = c['cluster_num']
            curr_seta = curr_setas[cluster_num]
            if curr_seta is None:
                print(f"❌ curr_seta not found for cluster_num: {cluster_num}")
                continue  # skip or handle this cluster

            if m_body_value is None:
                print("❌ ERROR: m_body_value is None")
                m_body_value = 0.01  # fallback default

            b = curr_seta + (m_body_value)

            if (a >= b or a <= b):
                new_seta = curr_seta - (m_body_value)
            else:
                new_seta = a
            
            r_value = centriods[cluster_num - 1]['r_setas']['trita']['r']

            # ✅ เก็บผลลัพธ์รอบนี้
            if repeat == 1:
                result_list.append({
                    "cluster_num": cluster_num,
                    "m": m_body_value,
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
    original_pixels = body.get("original_pixels", [])
    m_body_value = body.get("m_body_value", 0.01)
    print('m_body_value: ', m_body_value)

    list_of_clusters = []
    list_of_m = []

    for i in range(len(centriods)):
        # print(centriods[i]['r_setas'][r_seta_type]['seta'])
        list_of_clusters.append({
            "cluster_num": i + 1,
            "seta": centriods[i]['r_setas'][r_seta_type]['seta'],
            "r": centriods[i]['r_setas'][r_seta_type]['r']
        })
    
    # print(list_of_clusters)

    n = len(list_of_clusters)
    for k in range(n):
        m_k = ((list_of_clusters[k]['seta'] - list_of_clusters[k - 1]['seta']) ** 2) + ((list_of_clusters[(k + 1) % n]['seta'] - list_of_clusters[k]['seta']) ** 2)
        # print(M)
        list_of_m.append({
            "cluster_num": list_of_clusters[k]['cluster_num'],
            "cluster_seta": list_of_clusters[k]['seta'],
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
        remap_result = remap_for_trita(centriods, curr_setas, m_body_value, sorted_m_k, seta_prime_new_after, seta_prime_new_previous)
    else:
        remap_result = remap_for_prota_and_deutera(centriods, r_seta_type, m_body_value, curr_setas, sorted_m_k, seta_prime_new_after, seta_prime_new_previous)

    # remap_for_trita(curr_setas, sorted_m_k, seta_prime_new_after, seta_prime_new_previous)
    remap_payload_result = {
        "type": r_seta_type,
        "result": remap_result
    }

    u_confs = {
        "prota": 0.678,
        # "deutera": -1.217,
        "deutera": 0.54,
        "trita": 0.257
    }

    v_confs = {
        "prota": 0.501,
        # "deutera": 0.782,
        "deutera": 0.56,
        "trita": 0.0
    }

    translate_u_v = {
        "type": r_seta_type,
        "result": []
    }

    XYZ_TO_RGB_MATRIX = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    # cluster_to_new_seta = {entry['cluster_num']: entry['new_seta'] for entry in remap_payload_result}
    # cluster_to_new_seta = {entry['cluster_num']: entry['new_seta'] for entry in remap_payload_result['result']}
    cluster_to_new_seta = {
        entry['cluster_num'] - 1: entry['new_seta']  # ✅ Fix here
        for entry in remap_payload_result['result']
    }
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

        if new_seta is None or math.isnan(new_seta) or math.isinf(new_seta):
            print(f"❌ Skipping pixel at ({pixel_position['x']},{pixel_position['y']}) — invalid new_seta: {new_seta}")
            continue

        # print(f"\nPixel idx: {idx}")
        # print(f"  Cluster num: {cluster_num}")
        # print(f"  Pixel position: {pixel_position}")
        # print(f"  new_seta found: {new_seta}")

        if new_seta is not None:
            # ไปหา pixel ใน array_of_luv ที่ position ตรงกัน
            luv_pixel = array_of_luv[idx]
            if 'position' in luv_pixel and luv_pixel['position']['x'] == pixel_position['x'] and luv_pixel['position']['y'] == pixel_position['y']:
            # if luv_pixel['position']['x'] == pixel_position['x'] and luv_pixel['position']['y'] == pixel_position['y']:
                # เพิ่ม new_seta
                luv_pixel['new_seta'] = new_seta
                cluster_to_new_r.append(luv_pixel)
    
    list_of_cluster_r = [entry['r'] for entry in list_of_clusters]

    # print(list_of_cluster_r)

    r_cluster_mean = statistics.fmean(list_of_cluster_r)
    r_cluster_max = max(list_of_cluster_r)
    r_cluster_min = min(list_of_cluster_r)

    print(f"[DEBUG] cluster_to_new_seta keys: {list(cluster_to_new_seta.keys())[:5]} ...")
    print(f"[DEBUG] first 5 pixel cluster_nums: {[p['cluster_num'] for p in list_of_pixel_clusters[:5]]}")


    for i in range(len(cluster_to_new_r)):
        curr_pixel = cluster_to_new_r[i]
        u_prime = (u_confs[r_seta_type] + (curr_pixel['r'] * math.cos(curr_pixel['new_seta'])))
        v_prime = (v_confs[r_seta_type] + (curr_pixel['r'] * math.sin(curr_pixel['new_seta'])))

        curr_pixel['u_prime'] = u_prime
        curr_pixel['v_prime'] = v_prime

        b = 25
        curr_r_ij = curr_pixel['r']
        curr_l_ij = curr_pixel['l']

        denom = r_cluster_max - r_cluster_min
        if abs(denom) < 1e-5:
            denom = 1e-5  # prevent divide-by-zero

        l_prime = ((b * (curr_r_ij - r_cluster_mean)) / (r_cluster_max - r_cluster_min)) + curr_l_ij
        l_prime = max(1.0, l_prime)  # ✅ prevent invalid luminance

        curr_pixel['l_prime'] = l_prime

        # Convert L'U'V' to XYZ
        if (l_prime > 8):
            new_y = ((l_prime + 16) / 116) ** 3
        else:
            new_y = l_prime / 903.3
        
        # Ensure new_y is positive and properly scaled
        new_y = max(0, new_y)

        if abs(v_prime) < 1e-5:
            v_prime = 1e-5
        
        # Calculate new_x and new_z with proper scaling
        if v_prime != 0:  # Avoid division by zero
            # Scale u_prime and v_prime to proper ranges
            u_prime_scaled = u_prime * 1  # Scale down to prevent overflow
            v_prime_scaled = v_prime * 1
            
            new_x = ((9 * u_prime_scaled) / (4 * v_prime_scaled)) * new_y
            new_z = ((12 - (3 * u_prime_scaled) - (20 * v_prime_scaled)) / (4 * v_prime_scaled)) * new_y
        else:
            new_x = 0
            new_z = 0

        # Clamp XYZ values to reasonable ranges
        new_x = max(0, min(new_x, 1))
        new_y = max(0, min(new_y, 1))
        new_z = max(0, min(new_z, 1))

        curr_pixel['new_x'] = new_x
        curr_pixel['new_y'] = new_y
        curr_pixel['new_z'] = new_z

        # Convert XYZ to RGB using proper matrix multiplication
        xyz_vector = np.array([[new_x], [new_y], [new_z]])
        new_first_rgb = np.dot(XYZ_TO_RGB_MATRIX, xyz_vector)

        # Clamp values after matrix multiplication
        new_first_rgb = np.clip(new_first_rgb, 0.0, 1.0)

        first_r = float(new_first_rgb[0][0])
        first_g = float(new_first_rgb[1][0])
        first_b = float(new_first_rgb[2][0])

        first_r = max(0, first_r)
        first_g = max(0, first_g)
        first_b = max(0, first_b)

        # Gamma correction with proper scaling
        def gamma_correct(c):
            if c > 0.0031308:
                return 1.055 * (c ** (1 / 2.4)) - 0.055
            else:
                return 12.92 * c

        new_r = gamma_correct(first_r)
        new_g = gamma_correct(first_g)
        new_b = gamma_correct(first_b)

        # Final RGB clamping and scaling to 0-255 range
        final_r = min(max(new_r, 0.0), 1.0) * 255
        final_g = min(max(new_g, 0.0), 1.0) * 255
        final_b = min(max(new_b, 0.0), 1.0) * 255

        curr_pixel['new_rgb'] = {
            "r": int(final_r),
            "g": int(final_g),
            "b": int(final_b)
        }

    # Get image dimensions from the pixel data
    if cluster_to_new_r:
        # Find max x and y coordinates to determine image dimensions
        max_x = max(pixel['position']['x'] for pixel in cluster_to_new_r)
        max_y = max(pixel['position']['y'] for pixel in cluster_to_new_r)
        width = max_x + 1
        height = max_y + 1
        
        # Create a new image with the calculated dimensions
        new_image = Image.new('RGB', (width, height))
        
        # Create a dictionary to map positions to RGB values
        position_to_rgb = {}
        for pixel in cluster_to_new_r:
            if 'position' in pixel and 'new_rgb' in pixel:
                pos = (pixel['position']['x'], pixel['position']['y'])
                rgb = (
                    pixel['new_rgb']['r'],
                    pixel['new_rgb']['g'],
                    pixel['new_rgb']['b']
                )
                position_to_rgb[pos] = rgb
                # print(f"\n--- Pixel Debug ---")
                # print(f"Cluster #{cluster_num}")
                # print(f"new_seta: {new_seta}")
                # print(f"r: {curr_pixel['r']}, l: {curr_pixel['l']}, l': {l_prime}")
                # print(f"u': {u_prime}, v': {v_prime}")
                # print(f"XYZ = {new_x:.4f}, {new_y:.4f}, {new_z:.4f}")
                # print(f"Pre-Gamma RGB = ({first_r:.4f}, {first_g:.4f}, {first_b:.4f})")
                # print(f"Final RGB = ({final_r:.2f}, {final_g:.2f}, {final_b:.2f})")
        
        
        
        # Fill the image with the new RGB values
        for y in range(height):
            for x in range(width):
                if (x, y) in position_to_rgb:
                    new_image.putpixel((x, y), position_to_rgb[(x, y)])
        
        # Resize image if it's too large
        max_size = 800  # Maximum dimension
        if width > max_size or height > max_size:
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            new_image = new_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert the image to base64 with compression
        buffered = io.BytesIO()
        new_image.save(buffered, format="JPEG", quality=85)  # Use JPEG for better compression
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'pixelsData': cluster_to_new_r,
            'processedImage': img_str,
            # 'delta_e': delta_e.tolist(),
            # 'delta_e': delta_e_avg,
            'imageFormat': 'jpeg'  # Add format information
        }



# Download Img
@app.post('/download-image')
async def download_image(req: Request):
    body = await req.json()
    image_data = body.get("imageData")
    
    if not image_data:
        return {"error": "No image data provided"}
    
    # Decode base64 image data
    try:
        image_bytes = base64.b64decode(image_data)
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "attachment; filename=processed_image.jpg"
            }
        )
    except Exception as e:
        return {"error": str(e)}


# Delta e parts
def compute_intra_image_delta_e(image: Image.Image):
    width, height = image.size
    lab_list = []

    # เก็บค่า Lab ของทุกพิกเซล
    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            lab = convert_color(sRGBColor(r, g, b, is_upscaled=True), LabColor)
            lab_list.append([lab.lab_l, lab.lab_a, lab.lab_b])

    lab_array = np.array(lab_list)
    mean_lab = np.mean(lab_array, axis=0)

    # คำนวณ Delta E เทียบกับค่าเฉลี่ย
    mean_lab_array = np.tile(mean_lab, (lab_array.shape[0], 1))
    delta_e = deltaE_ciede2000(lab_array, mean_lab_array)
    delta_e_avg = float(np.mean(delta_e))

    return delta_e_avg

@app.post('/findDeltaE')
async def find_intra_delta_e(
    img1: UploadFile = File(...),
    img2: UploadFile = File(...)
):
    image1 = Image.open(io.BytesIO(await img1.read())).convert("RGB")
    image2 = Image.open(io.BytesIO(await img2.read())).convert("RGB")

    delta1 = compute_intra_image_delta_e(image1)
    delta2 = compute_intra_image_delta_e(image2)

    return {
        "intra_delta_e_image1": round(delta1, 4),
        "intra_delta_e_image2": round(delta2, 4),
        "msg": "ΔE ภายในภาพคำนวณสำเร็จ"
    }
