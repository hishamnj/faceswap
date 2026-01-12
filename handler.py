import runpod
import boto3
import cv2
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer

s3 = boto3.client("s3")

face_app = None
swapper = None
gfpgan = None

def init_models():
    global face_app, swapper, gfpgan

    if face_app is None:
        face_app = FaceAnalysis(providers=["CUDAExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))

    if swapper is None:
        swapper = get_model(
            "inswapper_128.onnx",
            download=True,
            providers=["CUDAExecutionProvider"]
        )

    if gfpgan is None:
        gfpgan = GFPGANer(
            model_path=None,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            device="cuda"
        )

def download(bucket, key, path):
    s3.download_file(bucket, key, path)

def face_swap(role_img, child_img):
    faces_role = face_app.get(role_img)
    faces_child = face_app.get(child_img)

    if not faces_role or not faces_child:
        raise Exception("Face not detected")

    return swapper.get(
        role_img,
        faces_role[0],
        faces_child[0],
        paste_back=True
    )

def handler(job):
    init_models()

    inp = job["input"]
    os.makedirs("/tmp", exist_ok=True)

    role_path = "/tmp/role.jpg"
    child_path = "/tmp/child.jpg"
    output_path = "/tmp/output.jpg"

    # Download role + child images from S3
    download(inp["role_bucket"], inp["role_key"], role_path)
    download(inp["child_bucket"], inp["child_key"], child_path)

    role_img = cv2.imread(role_path)
    child_img = cv2.imread(child_path)

    if role_img is None or child_img is None:
        raise Exception("Failed to load images")

    swapped = face_swap(role_img, child_img)

    restored, _, _ = gfpgan.enhance(
        swapped, has_aligned=False, only_center_face=True
    )

    cv2.imwrite(output_path, restored)

    out_key = f"result-{os.path.basename(inp['child_key'])}"
    s3.upload_file(output_path, inp["output_bucket"], out_key)

    return {
        "status": "done",
        "output": out_key
    }

runpod.serverless.start({"handler": handler})

