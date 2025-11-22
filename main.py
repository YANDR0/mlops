from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms, models
from PIL import Image
import io
import torch.nn as nn


CLASSES = ["cat", "dog"]
NUM_CLASSES = len(CLASSES)
model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, NUM_CLASSES)
)

model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()
app = FastAPI()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.get("/")
def root():
    return {"message": "API funcionando correctamente"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        predicted = torch.argmax(outputs, dim=1).item()

    label = CLASSES[predicted]
    return {"prediction": label}
