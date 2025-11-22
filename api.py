from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)


CLASSES = ["cat", "dog"]
NUM_CLASSES = len(CLASSES)

model = CNN(NUM_CLASSES)
model.load_state_dict(torch.load("best_catsdogs_model.pth", map_location="cpu"))
model.eval()

app = FastAPI()
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
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
        _, predicted = torch.max(outputs, 1)
        label = CLASSES[predicted.item()]

    return {"prediction": label}
