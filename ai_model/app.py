from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import json
import math
from fastapi.middleware.cors import CORSMiddleware

# 1. الإعدادات
with open("config.json") as f: CFG = json.load(f)
with open("label_map_v6.json") as f: label_map = json.load(f)

idx_to_label = {v:k for k,v in label_map.items()}
INPUT_SIZE, NUM_CLASSES, DEVICE = CFG["INPUT_SIZE"], len(label_map), "cpu"

# 2. تعريف الطبقات للنموذج
class PE(nn.Module):
    def __init__(self, d, T=30):
        super().__init__()
        pe = torch.zeros(T, d)
        pos = torch.arange(T).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class SignTransformerV6(nn.Module):
    def __init__(self, inp=INPUT_SIZE, d=256, h=8, L=4, ff=512, nc=NUM_CLASSES):
        super().__init__()
        self.hand_proj = nn.Linear(126, d)
        self.face_proj = nn.Linear(120, d)
        self.merge = nn.Linear(d*2, d)
        self.pos = PE(d)
        enc_l = nn.TransformerEncoderLayer(d, h, ff, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_l, L, norm=nn.LayerNorm(d))
        self.head = nn.Sequential(
            nn.Linear(d, d//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d//2, nc)
        )

    def forward(self, x):
        x = self.merge(torch.cat([self.hand_proj(x[:, :, :126]), self.face_proj(x[:, :, 126:])], dim=-1))
        x = self.enc(self.pos(x))
        x = (x.mean(1) + x.max(1).values) / 2
        return self.head(x)

# 3. تحميل النموذج
model = SignTransformerV6()
state_dict = torch.load("best_v6.pt", map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()

# 4. الـ API وإعدادات الأمان لـ Flutter
app = FastAPI(title="Sign Language Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionPayload(BaseModel):
    sequence: list  # المتوقع مصفوفة بالأبعاد (Frames, Features)

@app.get("/")
def home():
    return {"status": "Sign Language API is running successfully!"}

@app.post("/predict")
def predict(payload: PredictionPayload):
    try:
        raw_sequence = payload.sequence
        
        if not raw_sequence or len(raw_sequence) == 0:
            raise HTTPException(status_code=400, detail="Sequence container is empty.")
            
        processed_frames = []
        
        # تظبيط حماية الداتا: التأكد إن كل فريم رايح للموديل بـ 246 قيمة بالظبط
        for frame in raw_sequence:
            frame_array = np.array(frame, dtype=np.float32).flatten()
            
            # لو الفلاتر باعت الفريم ناقص أو مش مظبوط، بنكمل الباقي أصفار عشان الموديل ما يضربش
            if len(frame_array) < 246:
                padded_frame = np.zeros(246, dtype=np.float32)
                padded_frame[:len(frame_array)] = frame_array
                processed_frames.append(padded_frame)
            elif len(frame_array) > 246:
                processed_frames.append(frame_array[:246])
            else:
                processed_frames.append(frame_array)
                
        # الموديل مستني بالظبط 30 فريم، لو التيم باعت أقل أو أكتر بنظبطها لـ 30
        while len(processed_frames) < 30:
            processed_frames.append(np.zeros(246, dtype=np.float32))
        if len(processed_frames) > 30:
            processed_frames = processed_frames[:30]
            
        final_input = np.array([processed_frames], dtype=np.float32) # الأبعاد بقت (1, 30, 246)
        
      # تشغيل الموديل للتوقع (نسخة الـ Top-3 predictions الشيك)
        input_tensor = torch.tensor(final_input, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # حساب الاحتمالات لكل الكلمات
            probs = torch.softmax(outputs, dim=1)[0]
            
            # جلب أعلى 3 كلمات في نسبة الثقة
            top_probs, top_indices = torch.topk(probs, k=3)

        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            predictions.append({
                "word": idx_to_label.get(idx, "Unknown"),
                "confidence": round(prob * 100, 2)  # النسبة المئوية للثقة
            })

        return {
            "predictions": predictions,
            "status": "Success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")
