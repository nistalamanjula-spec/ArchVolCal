# Plan Takeoff – quantities and volume from architectural drawings

Upload a floor plan (PDF/PNG/JPG/TIFF) -> get every room's size, area and volume, plus slab concrete, as a table, annotated preview and CSV.

## Install
1. Install Tesseract OCR
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki (add to PATH)
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt install tesseract-ocr`
2. `pip install -r requirements.txt`
3. `python app.py` and open http://127.0.0.1:5000

## How it works
1. PyMuPDF renders the PDF at 300 dpi (images are used as-is).
2. Auto-rotation: tries 0/90/180/270 degrees and keeps the one where OCR finds the most dimension strings.
3. Tesseract (LSTM) reads words; they are merged into lines.
4. DBSCAN (scikit-learn) clusters lines into room labels (name + "A X B M" + "N SQ.M").
5. Regex parses the numbers; names are snapped to an architectural vocabulary; low-confidence labels get a second, upscaled OCR pass.
6. Cross-check: length x width vs the printed area. Disagreement is flagged "Review".
7. OpenCV measures the enclosed pixel region around each label to estimate drawing scale (when enough rooms are closed).
8. Volume = area x floor height. Slab concrete = area x thickness. Multiplied by number of floors.

## Limits
- Works best on plans that print dimensions/areas as text. It reads labels; it does not trace walls or infer sizes from geometry.
- Only page 1 of a PDF is read. Always verify rows marked "Review".
- Outputs are estimates for planning, not a substitute for a quantity surveyor's takeoff.

## Ideas to extend
Wall/brick/plaster quantities, door/window schedules, YOLO model trained on your own plans, multi-page PDFs, Excel export.
