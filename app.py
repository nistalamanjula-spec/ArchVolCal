"""Architectural quantity extractor - Flask server.   Run: python app.py  ->  http://127.0.0.1:5000"""
import os
import tempfile

from flask import Flask, jsonify, render_template, request

import extractor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose a plan to upload."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error="Upload a PDF, PNG, JPG, TIFF or WebP file."), 400
    try:
        height = float(request.form.get("height", 3.0))
        slab = float(request.form.get("slab", 0.15))
        floors = int(request.form.get("floors", 1))
    except ValueError:
        return jsonify(error="Height, slab thickness and floors must be numbers."), 400
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "plan" + ext)
        f.save(path)
        try:
            return jsonify(extractor.analyse(path, height, slab, max(1, floors)))
        except Exception as exc:  # surface a readable message to the UI
            return jsonify(error=str(exc)), 500


if __name__ == "__main__":
    app.run(debug=True)
