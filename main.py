"""
main.py - Main Application Entry Point
Flask web application for Diabetic Retinopathy Prediction System.

Run: python main.py
Then open: http://localhost:5000
"""

import os
import uuid
from datetime import datetime
from flask import (Flask, render_template, request, redirect, url_for,
                   jsonify, send_file, flash, session)
from werkzeug.utils import secure_filename

import database as db
from predict import predict_image, CLASS_NAMES
from utils.report_generator import generate_prediction_report

# ─── Flask Setup ──────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "dr_system_secret_key_2024"

@app.context_processor
def inject_globals():
    """Inject global variables into all templates."""
    return {"now": datetime.now()}

UPLOAD_FOLDER   = "uploads"
ALLOWED_EXT     = {"png", "jpg", "jpeg", "bmp"}
MAX_FILE_MB     = 16

app.config["UPLOAD_FOLDER"]    = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("reports",      exist_ok=True)
os.makedirs("sql",          exist_ok=True)


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT)


def unique_filename(filename):
    """Generate a unique filename to prevent collisions."""
    ext  = filename.rsplit(".", 1)[1].lower()
    stem = secure_filename(filename.rsplit(".", 1)[0])
    uid  = uuid.uuid4().hex[:8]
    return f"{stem}_{uid}.{ext}"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Home / Landing page."""
    stats = db.get_statistics()
    return render_template("home.html", stats=stats)


@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")


@app.route("/contact")
def contact():
    """Contact page."""
    return render_template("contact.html")


# ─── Prediction ───────────────────────────────────────────────────────────────

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """Prediction page — handles image upload and runs the model."""
    if request.method == "GET":
        return render_template("predict.html")

    # ── POST: process uploaded image ──
    # 1. Validate file
    if "retinal_image" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("predict_page"))

    file = request.files["retinal_image"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("predict_page"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload PNG, JPG, JPEG, or BMP.", "error")
        return redirect(url_for("predict_page"))

    # 2. Save uploaded image
    filename  = unique_filename(file.filename)
    filepath  = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # 3. Get or create patient
    patient_name   = request.form.get("patient_name", "Anonymous").strip() or "Anonymous"
    patient_age    = request.form.get("patient_age", "")
    patient_gender = request.form.get("patient_gender", "")
    patient_email  = request.form.get("patient_email", f"anon_{uuid.uuid4().hex[:6]}@dr.local")
    patient_phone  = request.form.get("patient_phone", "")
    patient_addr   = request.form.get("patient_address", "")

    try:
        age = int(patient_age) if patient_age else None
    except ValueError:
        age = None

    patient_id = db.insert_patient(patient_name, age, patient_gender,
                                    patient_email, patient_phone, patient_addr)

    # 4. Save image record
    image_id = db.insert_image(patient_id, filename, filepath)

    # 5. Run prediction
    try:
        result = predict_image(filepath)
    except FileNotFoundError:
        flash("Model not found. Please train the model first (python train_model.py).",
              "warning")
        # Return a demo result so the UI still renders
        result = _demo_prediction()

    # 6. Store prediction in DB
    pred_id = db.insert_prediction(
        patient_id   = patient_id,
        image_id     = image_id,
        predicted_class = result["predicted_class"],
        confidence      = result["confidence"],
        severity_level  = result["severity"],
        advice          = result["advice"]
    )

    # 7. Render results page
    return render_template(
        "result.html",
        result      = result,
        patient_name= patient_name,
        patient_id  = patient_id,
        image_id    = image_id,
        pred_id     = pred_id,
        filename    = filename,
    )


def _demo_prediction():
    """Return a demo prediction when model is not trained yet."""
    import random
    classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    keys    = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
    idx     = random.randint(0, 4)
    from predict import CLASS_LABELS, SEVERITY_MAP, ADVICE_MAP, RISK_MAP
    key = keys[idx]
    rl, rc = RISK_MAP[key]
    probs = {c: round(random.uniform(0, 100), 2) for c in classes}
    total = sum(probs.values())
    probs = {c: round(v / total * 100, 2) for c, v in probs.items()}
    return {
        "predicted_class": CLASS_LABELS[key],
        "class_key":       key,
        "confidence":      probs[CLASS_LABELS[key]],
        "severity":        SEVERITY_MAP[key],
        "advice":          ADVICE_MAP[key],
        "risk_level":      rl,
        "risk_color":      rc,
        "all_probs":       probs,
    }


# ─── Patient Records ──────────────────────────────────────────────────────────

@app.route("/patients")
def patients_page():
    """Patient records listing page."""
    query   = request.args.get("q", "").strip()
    if query:
        patients = db.search_patients(query)
    else:
        patients = db.get_all_patients()
    return render_template("patients.html", patients=patients, query=query)


@app.route("/patients/<int:patient_id>")
def patient_detail(patient_id):
    """Detailed view of a single patient with prediction history."""
    patient     = db.get_patient_by_id(patient_id)
    predictions = db.get_predictions_by_patient(patient_id)
    if not patient:
        flash("Patient not found.", "error")
        return redirect(url_for("patients_page"))
    return render_template("patient_detail.html",
                            patient=patient, predictions=predictions)


@app.route("/patients/delete/<int:patient_id>", methods=["POST"])
def delete_patient(patient_id):
    """Delete a patient and all related data."""
    db.delete_patient(patient_id)
    flash("Patient record deleted successfully.", "success")
    return redirect(url_for("patients_page"))


# ─── Prediction History ───────────────────────────────────────────────────────

@app.route("/history")
def history_page():
    """Prediction history page."""
    predictions = db.get_all_predictions()
    return render_template("history.html", predictions=predictions)


# ─── Report Generation ────────────────────────────────────────────────────────

@app.route("/report/<int:pred_id>")
def generate_report(pred_id):
    """Generate and download a PDF report for a prediction."""
    prediction = db.get_prediction_by_id(pred_id)
    if not prediction:
        flash("Prediction not found.", "error")
        return redirect(url_for("history_page"))

    # Find the image
    image_path = None
    img_filename = prediction.get("filename")
    if img_filename:
        candidate = os.path.join(UPLOAD_FOLDER, img_filename)
        if os.path.exists(candidate):
            image_path = candidate

    # Generate PDF
    filepath, filename = generate_prediction_report(prediction, image_path)

    # Save report record
    db.insert_report(
        patient_id    = prediction["patient_id"],
        prediction_id = pred_id,
        report_filename = filename,
        report_path   = filepath
    )

    return send_file(filepath, as_attachment=True, download_name=filename)


# ─── API Endpoints (AJAX) ─────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    """Return dashboard statistics as JSON."""
    return jsonify(db.get_statistics())


@app.route("/api/predict_ajax", methods=["POST"])
def api_predict_ajax():
    """Lightweight AJAX prediction endpoint (for JavaScript fetch)."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = unique_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        result = predict_image(filepath)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify(_demo_prediction())


# ─── Static file helper ───────────────────────────────────────────────────────

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded images."""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(413)
def too_large(e):
    flash(f"File too large. Maximum size is {MAX_FILE_MB} MB.", "error")
    return redirect(url_for("predict_page"))


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Diabetic Retinopathy Prediction System")
    print("  Open: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
