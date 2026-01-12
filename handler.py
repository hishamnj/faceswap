from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn
import uuid
import boto3
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer

app = FastAPI()

# AWS S3 Client
s3 = boto3.client("s3")

# Global model variables
face_app = None
swapper = None
gfpgan = None

class FaceSwapRequest(BaseModel):
    role_bucket: str
    role_key: str
    child_bucket: str
    child_key: str
    output_bucket: str

def init_models():
    global face_app, swapper, gfpgan

    if face_app is None:
        print("Initializing FaceAnalysis...")
        face_app = FaceAnalysis(providers=["CUDAExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))

    if swapper is None:
        print("Initializing InSwapper...")
        swapper = get_model(
            "inswapper_128.onnx",
            download=True,
            providers=["CUDAExecutionProvider"]
        )

    if gfpgan is None:
        print("Initializing GFPGAN...")
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

@app.on_event("startup")
async def startup_event():
    init_models()

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

@app.post("/generate")
async def generate(request: FaceSwapRequest):
    request_id = str(uuid.uuid4())
    role_path = f"/tmp/role_{request_id}.jpg"
    child_path = f"/tmp/child_{request_id}.jpg"
    output_path = f"/tmp/output_{request_id}.jpg"
    
    try:
        os.makedirs("/tmp", exist_ok=True)

        # Download role + child images from S3
        download(request.role_bucket, request.role_key, role_path)
        download(request.child_bucket, request.child_key, child_path)

        role_img = cv2.imread(role_path)
        child_img = cv2.imread(child_path)

        if role_img is None or child_img is None:
            raise HTTPException(status_code=400, detail="Failed to load images")

        swapped = face_swap(role_img, child_img)

        restored, _, _ = gfpgan.enhance(
            swapped, has_aligned=False, only_center_face=True
        )

        cv2.imwrite(output_path, restored)

        out_key = f"result-{request_id}-{os.path.basename(request.child_key)}"
        s3.upload_file(output_path, request.output_bucket, out_key)

        return {
            "status": "done",
            "output": out_key
        }
    except Exception as e:
        print(f"Error during face swap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup local files
        for p in [role_path, child_path, output_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass

if __name__ == "__main__":
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port)
