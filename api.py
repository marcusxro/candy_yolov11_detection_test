from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any
import io

app = FastAPI(
    title="candy detection",
    description="API for detecting objects in images using YOLO",
    version="1.0.0"
)

model = None
labels = None

@app.on_event("startup")
async def startup_event():
    global model, labels
    try:
        model = YOLO('model.pt')  
        labels = model.names
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.post("/predict", response_model=Dict[str, Any])
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Detect objects in an uploaded image and return the most confident detection.
    
    Parameters:
    - file: UploadFile containing the image
    
    Returns:
    - JSON with class name and confidence percentage
    """

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image")

    try:
    
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        results = model(img)
        detections = results[0].boxes

        if len(detections) == 0:
            return {"message": "No objects detected"}


        best_detection = max(detections, key=lambda x: x.conf.item())
        
        return {
            "class": labels[int(best_detection.cls.item())],
            "confidence": f"{float(best_detection.conf.item()) * 100:.2f}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
