import os
import re
import pickle
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import TransformerClassifier

# ── Config ──────────────────────────────────────────────
DEVICE   = "cpu"
MAX_LEN  = 256
MODEL_PATH = "best_transformer.pth"
VOCAB_PATH = "vocab.pkl"

# ── Load vocab ──────────────────────────────────────────
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

# ── Load model ──────────────────────────────────────────
model = TransformerClassifier()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ── FastAPI app ─────────────────────────────────────────
app = FastAPI(title="Transformer Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ──────────────────────────────────────
class TextRequest(BaseModel):
    text: str

# ── Tokenizer helper ────────────────────────────────────
def encode(text: str):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [vocab.word2idx.get("<CLS>", 2)]
    for w in text.split():
        tokens.append(vocab.word2idx.get(w, vocab.word2idx.get("<UNK>", 1)))
    tokens = tokens[:MAX_LEN]
    tokens += [0] * (MAX_LEN - len(tokens))
    return tokens

# ── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Transformer API is running"}

@app.post("/predict")
def predict(req: TextRequest):
    if not req.text.strip():
        return {"error": "Empty text"}

    ids    = torch.tensor([encode(req.text)], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
        probs  = F.softmax(logits, dim=-1).squeeze()

    label     = probs.argmax().item()
    sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
    confidence = round(probs[label].item() * 100, 2)

    return {
        "sentiment"  : sentiment,
        "confidence" : f"{confidence}%",
        "positive"   : f"{round(probs[1].item()*100, 2)}%",
        "negative"   : f"{round(probs[0].item()*100, 2)}%",
  }
