from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import json
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compare", methods=["POST"])
def compare():
    # Save uploaded files
    golden_file = request.files["golden"]
    test_file = request.files["test"]
    json_file = request.files["json"]

    golden_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(golden_file.filename))
    test_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(test_file.filename))
    json_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(json_file.filename))

    golden_file.save(golden_path)
    test_file.save(test_path)
    json_file.save(json_path)

    # Load images
    golden = cv2.imread(golden_path)
    test = cv2.imread(test_path)

    if golden is None or test is None:
        return "❌ Error loading images"

    # Load JSON
    with open(json_path, "r") as f:
        components = json.load(f)

    img_h, img_w = golden.shape[:2]
    missing_components = []

    for comp in components:
        name = comp["name"]
        cx, cy, bw, bh = comp["x"] * img_w, comp["y"] * img_h, comp["w"] * img_w, comp["h"] * img_h

        x, y, w, h = int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)

        ref_crop = golden[y:y+h, x:x+w]
        test_crop = test[y:y+h, x:x+w]

        if ref_crop.size == 0 or test_crop.size == 0:
            continue

        ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(test_crop, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(ref_gray, test_gray)
        score = np.sum(diff)

        if score > 50000:
            missing_components.append(name)
            cv2.rectangle(test, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(test, f"Missing: {name}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.rectangle(test, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(test, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save result
    result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
    cv2.imwrite(result_path, test)

    return render_template("result.html", result_image="result.jpg", missing=missing_components)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename="uploads/" + filename))


if __name__ == "__main__":
    app.run(debug=True)
