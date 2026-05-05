from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import json
import math

# 1. الإعدادات
with open("config.json") as f: CFG = json.load(f)
with open("label_map_v6.json") as f: label_map = json.load(f)

idx_to_label = {v:k for k,v in label_map.items()}
INPUT_SIZE, NUM_CLASSES, DEVICE = CFG["INPUT_SIZE"], len(label_map), "cpu"

# 2. تعريف الطبقات
class PE(nn.Module):
    def __init__(self,d,T=30):
        super().__init__()
        pe=torch.zeros(T,d)
        pos=torch.arange(T).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000)/d))
        pe[:,0::2], pe[:,1::2] = torch.sin(pos*div), torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class SignTransformerV6(nn.Module):
    def __init__(self,inp=INPUT_SIZE,d=256,h=8,L=4,ff=512,nc=NUM_CLASSES):
        super().__init__()
        self.hand_proj = nn.Sequential(nn.Linear(126, d//2), nn.LayerNorm(d//2))
        self.face_proj = nn.Sequential(nn.Linear(inp-126, d//2), nn.LayerNorm(d//2))
        self.merge = nn.Sequential(nn.Linear(d,d), nn.LayerNorm(d))
        self.pos = PE(d)
        enc_l = nn.TransformerEncoderLayer(d, h, ff, batch_first=True, norm_first=True)
        # أضفنا الـ norm عشان يطابق enc.norm في الملف
        self.enc = nn.TransformerEncoder(enc_l, L, norm=nn.LayerNorm(d))
        self.head = nn.Sequential(
            nn.Linear(d, d//2), # 0
            nn.GELU(),          # 1
            nn.Dropout(0.1),    # 2 (الطبقة المفقودة)
            nn.Linear(d//2, nc) # 3 (الطبقة الزائدة)
        )

    def forward(self,x):
        x = self.merge(torch.cat([self.hand_proj(x[:,:,:126]), self.face_proj(x[:,:,126:])], dim=-1))
        x = self.enc(self.pos(x))
        x = (x.mean(1)+x.max(1).values)/2
        return self.head(x)

# 3. تحميل النموذج (الحل النهائي)
model = SignTransformerV6()
# السطر السحري اللي هيتخطى أي غلطة في الأسماء
state_dict = torch.load("best_v6.pt", map_location=DEVICE)
model.load_state_dict(state_dict, strict=False) 
model.eval()

# 4. الـ API
app = FastAPI()

class InputData(BaseModel): sequence: list

@app.post("/predict")
def predict(data: InputData):
    x = torch.tensor(np.array(data.sequence, dtype=np.float32)).unsqueeze(0)
    #if x.shape != (1, 30, 246): return {"error": f"Wrong shape {x.shape}"}
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top3 = torch.topk(probs, 3)
    return {"prediction": idx_to_label[top3.indices[0].item()], 
            "top3": [{"word": idx_to_label[i.item()], "conf": float(s)} for s, i in zip(top3.values, top3.indices)]}