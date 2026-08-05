"""
utils/report_generator.py
PDF report generation using ReportLab for Diabetic Retinopathy System.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage,
                                 HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Color scheme
DARK_BLUE  = colors.HexColor("#0a2463")
TEAL       = colors.HexColor("#1e88e5")
LIGHT_GRAY = colors.HexColor("#f0f4f8")
SUCCESS    = colors.HexColor("#2e7d32")
WARNING    = colors.HexColor("#f57c00")
DANGER     = colors.HexColor("#c62828")


def severity_color(severity):
    """Return color based on DR severity."""
    mapping = {
        "No DR":          SUCCESS,
        "Mild":           colors.HexColor("#81c784"),
        "Moderate":       WARNING,
        "Severe":         colors.HexColor("#e64a19"),
        "Proliferative DR": DANGER,
    }
    return mapping.get(severity, DARK_BLUE)


def generate_prediction_report(prediction_data, image_path=None):
    """
    Generate a PDF report for a single prediction.
    prediction_data: dict with keys from prediction_history + patient info.
    Returns the path to the generated PDF.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_id = prediction_data.get("patient_id", "unknown")
    filename = f"report_patient{patient_id}_{timestamp}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    story = []

    # --- Header ---
    header_style = ParagraphStyle(
        "Header", fontSize=20, textColor=DARK_BLUE,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6
    )
    sub_style = ParagraphStyle(
        "Sub", fontSize=11, textColor=TEAL,
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4
    )
    normal = ParagraphStyle(
        "Normal2", fontSize=10, textColor=colors.black,
        fontName="Helvetica", spaceAfter=4
    )
    bold_label = ParagraphStyle(
        "BoldLabel", fontSize=10, textColor=DARK_BLUE,
        fontName="Helvetica-Bold", spaceAfter=2
    )

    story.append(Paragraph("🏥 Diabetic Retinopathy Prediction Report", header_style))
    story.append(Paragraph("AI-Powered Retinal Disease Detection System", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL))
    story.append(Spacer(1, 0.4*cm))

    # --- Report metadata ---
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    story.append(Paragraph(f"Report Generated: {now}", normal))
    story.append(Paragraph(
        f"Report ID: RPT-{prediction_data.get('prediction_id', 'N/A'):05d}", normal))
    story.append(Spacer(1, 0.3*cm))

    # --- Patient Information ---
    story.append(Paragraph("Patient Information", bold_label))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))
    story.append(Spacer(1, 0.2*cm))

    patient_data = [
        ["Patient Name", prediction_data.get("full_name", "N/A")],
        ["Patient ID",   str(prediction_data.get("patient_id", "N/A"))],
        ["Age",          str(prediction_data.get("age", "N/A"))],
        ["Gender",       prediction_data.get("gender", "N/A")],
        ["Email",        prediction_data.get("email", "N/A")],
    ]

    pt_table = Table(patient_data, colWidths=[5*cm, 12*cm])
    pt_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), LIGHT_GRAY),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("PADDING",     (0, 0), (-1, -1), 6),
    ]))
    story.append(pt_table)
    story.append(Spacer(1, 0.4*cm))

    # --- Prediction Results ---
    story.append(Paragraph("Prediction Results", bold_label))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))
    story.append(Spacer(1, 0.2*cm))

    predicted_class = prediction_data.get("predicted_class", "N/A")
    confidence = prediction_data.get("confidence", 0)
    severity = prediction_data.get("severity_level", "N/A")
    s_color = severity_color(predicted_class)

    result_style = ParagraphStyle(
        "Result", fontSize=16, textColor=s_color,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6
    )
    story.append(Paragraph(f"Detected: {predicted_class}", result_style))
    story.append(Paragraph(
        f"Confidence: {confidence:.1f}%  |  Severity: {severity}",
        ParagraphStyle("Conf", fontSize=12, textColor=TEAL,
                       fontName="Helvetica", alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 0.3*cm))

    # Confidence bar (visual table trick)
    bar_filled = int(confidence / 5)   # out of 20 cells
    bar_empty  = 20 - bar_filled
    bar_data = [["Confidence Level"] + ["█"] * bar_filled + ["░"] * bar_empty]
    bar_table = Table(bar_data, colWidths=[4*cm] + [0.6*cm] * 20)
    bar_table.setStyle(TableStyle([
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (1, 0), (bar_filled, 0), s_color),
        ("TEXTCOLOR", (bar_filled+1, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME",  (0, 0), (0, 0), "Helvetica-Bold"),
        ("PADDING",   (0, 0), (-1, -1), 2),
    ]))
    story.append(bar_table)
    story.append(Spacer(1, 0.4*cm))

    # --- Retinal Image (if available) ---
    if image_path and os.path.exists(image_path):
        story.append(Paragraph("Analyzed Retinal Image", bold_label))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))
        story.append(Spacer(1, 0.2*cm))
        try:
            img = RLImage(image_path, width=8*cm, height=6*cm)
            story.append(img)
        except Exception:
            pass
        story.append(Spacer(1, 0.3*cm))

    # --- Medical Advice ---
    story.append(Paragraph("Medical Advice & Recommendations", bold_label))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))
    story.append(Spacer(1, 0.2*cm))

    advice = prediction_data.get("advice", get_default_advice(predicted_class))
    story.append(Paragraph(advice, normal))
    story.append(Spacer(1, 0.4*cm))

    # --- About DR Stages ---
    story.append(Paragraph("About Diabetic Retinopathy Stages", bold_label))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))
    story.append(Spacer(1, 0.2*cm))

    stages_info = [
        ["Stage", "Description", "Action Required"],
        ["No DR",
         "No signs of diabetic retinopathy detected.",
         "Annual screening"],
        ["Mild",
         "Small areas of balloon-like swelling in retinal vessels.",
         "Monitor every 9 months"],
        ["Moderate",
         "Blockage of some blood vessels nourishing retina.",
         "Monitor every 6 months"],
        ["Severe",
         "Many blocked blood vessels, retina signals body to grow new.",
         "Consult specialist within 1 month"],
        ["Proliferative DR",
         "Advanced stage — new abnormal blood vessels grow on retina.",
         "IMMEDIATE specialist referral"],
    ]

    s_table = Table(stages_info, colWidths=[3.5*cm, 8*cm, 5.5*cm])
    s_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("PADDING",     (0, 0), (-1, -1), 5),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(s_table)
    story.append(Spacer(1, 0.5*cm))

    # --- Disclaimer ---
    disclaimer_style = ParagraphStyle(
        "Disclaimer", fontSize=8, textColor=colors.grey,
        fontName="Helvetica-Oblique", alignment=TA_CENTER
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system for screening purposes only. "
        "It is NOT a substitute for professional medical diagnosis. "
        "Please consult a qualified ophthalmologist for clinical evaluation.",
        disclaimer_style
    ))

    doc.build(story)
    return filepath, filename


def get_default_advice(predicted_class):
    """Return default medical advice based on DR class."""
    advice_map = {
        "No DR": (
            "No signs of diabetic retinopathy were detected. Continue maintaining "
            "good blood sugar control and schedule annual eye examinations. "
            "Keep HbA1c levels below 7%, manage blood pressure, and maintain a healthy lifestyle."
        ),
        "Mild": (
            "Mild diabetic retinopathy has been detected. Schedule a follow-up eye exam "
            "within 9 months. Improve blood sugar and blood pressure control. "
            "Consult your primary care physician to review diabetes management plan."
        ),
        "Moderate": (
            "Moderate diabetic retinopathy detected. Consult an ophthalmologist within "
            "3–6 months. Strict glycemic control is critical. Consider laser treatment "
            "options with your specialist. Regular monitoring every 6 months is recommended."
        ),
        "Severe": (
            "Severe diabetic retinopathy detected. URGENT ophthalmology referral required "
            "within 1 month. Risk of vision loss is significant. Laser photocoagulation or "
            "anti-VEGF injections may be necessary. Do not delay treatment."
        ),
        "Proliferative DR": (
            "CRITICAL: Proliferative diabetic retinopathy detected. IMMEDIATE referral to "
            "a retinal specialist is required. There is high risk of severe vision loss or "
            "blindness. Treatments such as vitrectomy, laser surgery, or anti-VEGF therapy "
            "must be discussed urgently with a specialist."
        ),
    }
    return advice_map.get(predicted_class,
                          "Please consult an ophthalmologist for further evaluation.")
