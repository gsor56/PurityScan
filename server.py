"""PurityScan server: stores labelled samples, trains a k-nearest-neighbours
classifier on two features (d = direct-beam drop %, s = scatter rise %),
and answers /predict. Run: pip install -r requirements.txt && python server.py"""
import csv, os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

app = Flask(__name__)
CORS(app)  # lets your Vercel page call this API
DATA = os.path.join(os.path.dirname(__file__), "samples.csv")
WARN, FAIL = 12.0, 22.0
model = None
accuracy = None


def load():
    if not os.path.exists(DATA):
        return [], []
    X, y = [], []
    with open(DATA) as f:
        for r in csv.DictReader(f):
            X.append([float(r["d"]), float(r["s"])])
            y.append(r["label"])
    return X, y


def rule(d, s):
    dev = max(abs(d), abs(s))
    kind = "scattering" if abs(s) > abs(d) else "absorption"
    if dev < WARN:
        return "Matches reference", dev, None
    return ("Suspect" if dev < FAIL else "Different from reference"), dev, kind


def train():
    global model, accuracy
    X, y = load()
    counts = {c: y.count(c) for c in set(y)}
    if len(counts) < 2 or min(counts.values()) < 3:
        model, accuracy = None, None
        return {"trained": False, "reason": "Need at least 2 labels with 3+ samples each", "counts": counts}
    k = min(5, min(counts.values()))
    m = KNeighborsClassifier(n_neighbors=k).fit(np.array(X), y)
    cv = min(5, min(counts.values()))
    accuracy = float(cross_val_score(KNeighborsClassifier(n_neighbors=k), np.array(X), y, cv=cv).mean())
    model = m
    return {"trained": True, "accuracy": round(accuracy, 3), "k": k, "counts": counts}


@app.get("/health")
def health():
    X, y = load()
    return jsonify(ok=True, samples=len(y), trained=model is not None, accuracy=accuracy)


@app.post("/sample")
def sample():
    j = request.get_json(force=True)
    new = not os.path.exists(DATA)
    with open(DATA, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "d", "s", "s1", "s2", "b1", "b2"])
        w.writerow([j["label"], j["d"], j["s"], j.get("s1", ""), j.get("s2", ""), j.get("b1", ""), j.get("b2", "")])
    return jsonify(ok=True)


@app.post("/train")
def do_train():
    return jsonify(train())


@app.post("/predict")
def predict():
    j = request.get_json(force=True)
    d, s = float(j["d"]), float(j["s"])
    verdict, dev, kind = rule(d, s)
    out = {"verdict": verdict, "deviation": round(dev, 1), "type": kind, "label": None}
    if model is not None:
        proba = model.predict_proba([[d, s]])[0]
        i = int(np.argmax(proba))
        out["label"] = str(model.classes_[i])
        out["confidence"] = round(float(proba[i]), 2)
        out["model_accuracy"] = round(accuracy, 3)
    return jsonify(out)


if __name__ == "__main__":
    train()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
