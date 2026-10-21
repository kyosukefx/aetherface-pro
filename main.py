
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import torch
import facer

app = FastAPI(title="AetherFace Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"
detector = None
startup_error = None

@app.on_event("startup")
def load_detector():
    global detector, startup_error
    try:
        detector = facer.face_detector(
            "retinaface/mobilenet",
            device=device
        )
    except Exception as exc:
        startup_error = str(exc)

@app.get("/")
def home():
    return {
        "app": "AetherFace Pro",
        "status": "online",
        "detector_loaded": detector is not None
    }

@app.get("/health")
def health():
    return {
        "api": "online",
        "detector_loaded": detector is not None,
        "error": startup_error
    }

@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Detector indisponível: " + str(startup_error)
        )

    if file.content_type not in (
        "image/jpeg", "image/png", "image/webp"
    ):
        raise HTTPException(
            status_code=400,
            detail="Envie uma imagem JPEG, PNG ou WEBP."
        )

    try:
        content = await file.read()

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="A imagem deve ter no máximo 10 MB."
            )

        image = Image.open(io.BytesIO(content)).convert("RGB")
        image = image.resize(
            (min(image.width, 1280), min(image.height, 1280))
        )
        array = np.array(image)
        tensor = (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .float()
            .to(device)
        )

        with torch.no_grad():
            result = detector(tensor.unsqueeze(0))

        faces = result.get("faces")
        count = len(faces) if faces is not None else 0

        return {
            "success": True,
            "faces_detected": count,
            "message": (
                "Rostos detectados."
                if count else "Nenhum rosto detectado."
            )
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Falha ao analisar a imagem: " + str(exc)
        )
