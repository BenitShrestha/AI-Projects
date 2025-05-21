from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from inference import predict_sketch
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Sketchify backend is running!"}

@app.post("/sketch")
async def sketchify(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    sketch = predict_sketch(image)

    # Convert to bytes
    buffer = io.BytesIO()
    sketch.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")