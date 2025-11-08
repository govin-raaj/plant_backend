from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import json
import os


app = FastAPI(title="🌿 Plant Species Classifier API with Info")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "./best.pt"
model = YOLO(model_path)
model.to(device)

plant_info_path = "./plant_info.json"
if os.path.exists(plant_info_path):
    with open(plant_info_path, "r", encoding="utf-8") as f:
        plant_info = json.load(f)
    print("✅ Loaded plant_info.json successfully.")
else:
    plant_info = {}
    print("⚠️ plant_info.json not found. Continuing without extra info.")

class_names = ['African Violet (Saintpaulia ionantha)', 'Aloe Vera', 'Begonia (Begonia spp.)', 
               'Birds Nest Fern (Asplenium nidus)', 'Boston Fern (Nephrolepis exaltata)', 'Calathea',
                 'Cast Iron Plant (Aspidistra elatior)', 'Chinese Money Plant (Pilea peperomioides)', 
                 'Christmas Cactus (Schlumbergera bridgesii)', 'Dracaena', 'Elephant Ear (Alocasia spp.)', 
                 'English Ivy (Hedera helix)', 'Hyacinth (Hyacinthus orientalis)', 'Jade plant (Crassula ovata)', 
                 'Money Tree (Pachira aquatica)', 'Orchid', 'Parlor Palm (Chamaedorea elegans)', 'Peace lily', 
                 'Poinsettia (Euphorbia pulcherrima)', 'Polka Dot Plant (Hypoestes phyllostachya)', 'Pothos (Ivy arum)', 
                 'Rubber Plant (Ficus elastica)', 'Schefflera', 'Snake plant (Sanseviera)', 'Tradescantia', 'Tulip']


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        results = model.predict(source=np.array(img), imgsz=224, device=device, verbose=False)

        probs = results[0].probs.data.cpu().numpy()
        class_index = int(np.argmax(probs))
        pred_class = class_names[class_index] if class_index < len(class_names) else "Unknown"
        confidence = float(np.max(probs))

        info = plant_info.get(pred_class, {"message": "No detailed info available for this plant."})

        return JSONResponse(content={
            "predicted_class": pred_class,
            "confidence": round(confidence * 100, 2),
            "info": info
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/plants")
async def get_all_plants():

    plants_list = []
    for plant, details in plant_info.items():
        plants_list.append({
            "name": plant,
            "description": details.get("description", "No description available."),
            "growth": details.get("growth", {}),
            "diseases": details.get("diseases", {})
        })
    return {"count": len(plants_list), "plants": plants_list}


@app.get("/")
async def root():
    return {
        "message": "🌿 Plant Classification API with Info and CORS is running!",
        "endpoints": {
            "/predict": "POST an image to classify",
            "/plants": "GET all plant info"
        }
    }
"uvicorn fast_api:app --reload --host 0.0.0.0 --port 8000"
