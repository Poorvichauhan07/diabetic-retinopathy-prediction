"""
database.py - SQL Database Module
Handles all database operations for the Diabetic Retinopathy System
"""

import sqlite3
import os
from datetime import datetime

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), "sql", "retinopathy.db")


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enables dict-like row access
    return conn


def initialize_database():
    """
    Create all required tables if they don't exist.
    Tables: patients, prediction_history, uploaded_images, reports
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # TABLE 1: patients
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name    TEXT    NOT NULL,
            age          INTEGER,
            gender       TEXT,
            email        TEXT    UNIQUE,
            phone        TEXT,
            address      TEXT,
            created_at   TEXT    DEFAULT (datetime('now')),
            updated_at   TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ------------------------------------------------------------------
    # TABLE 2: uploaded_images
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_images (
            image_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id   INTEGER,
            filename     TEXT    NOT NULL,
            filepath     TEXT    NOT NULL,
            upload_time  TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    # ------------------------------------------------------------------
    # TABLE 3: prediction_history
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            prediction_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id        INTEGER,
            image_id          INTEGER,
            predicted_class   TEXT    NOT NULL,
            confidence        REAL    NOT NULL,
            severity_level    TEXT,
            advice            TEXT,
            prediction_time   TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (image_id)   REFERENCES uploaded_images(image_id)
        )
    """)

    # ------------------------------------------------------------------
    # TABLE 4: reports
    # ------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id        INTEGER,
            prediction_id     INTEGER,
            report_filename   TEXT,
            report_path       TEXT,
            generated_at      TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id)   REFERENCES patients(patient_id),
            FOREIGN KEY (prediction_id) REFERENCES prediction_history(prediction_id)
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


# ===========================================================
# PATIENT OPERATIONS
# ===========================================================

def insert_patient(full_name, age, gender, email, phone, address):
    """Insert a new patient record. Returns the new patient_id."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO patients (full_name, age, gender, email, phone, address)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (full_name, age, gender, email, phone, address))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        # Email already exists — fetch existing patient
        cursor.execute("SELECT patient_id FROM patients WHERE email = ?", (email,))
        row = cursor.fetchone()
        return row["patient_id"] if row else None
    finally:
        conn.close()


def get_all_patients():
    """Return a list of all patients."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_patient_by_id(patient_id):
    """Return a single patient record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def search_patients(query):
    """Search patients by name, email, or phone."""
    conn = get_connection()
    cursor = conn.cursor()
    like = f"%{query}%"
    cursor.execute("""
        SELECT * FROM patients
        WHERE full_name LIKE ? OR email LIKE ? OR phone LIKE ?
        ORDER BY created_at DESC
    """, (like, like, like))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_patient(patient_id, full_name, age, gender, email, phone, address):
    """Update patient details."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE patients
        SET full_name=?, age=?, gender=?, email=?, phone=?, address=?,
            updated_at=datetime('now')
        WHERE patient_id=?
    """, (full_name, age, gender, email, phone, address, patient_id))
    conn.commit()
    conn.close()


def delete_patient(patient_id):
    """Delete a patient and all related records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reports WHERE patient_id=?", (patient_id,))
    cursor.execute("DELETE FROM prediction_history WHERE patient_id=?", (patient_id,))
    cursor.execute("DELETE FROM uploaded_images WHERE patient_id=?", (patient_id,))
    cursor.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
    conn.commit()
    conn.close()


# ===========================================================
# IMAGE OPERATIONS
# ===========================================================

def insert_image(patient_id, filename, filepath):
    """Save uploaded image metadata. Returns image_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO uploaded_images (patient_id, filename, filepath)
        VALUES (?, ?, ?)
    """, (patient_id, filename, filepath))
    conn.commit()
    image_id = cursor.lastrowid
    conn.close()
    return image_id


# ===========================================================
# PREDICTION OPERATIONS
# ===========================================================

def insert_prediction(patient_id, image_id, predicted_class,
                      confidence, severity_level, advice):
    """Store a prediction result. Returns prediction_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_history
            (patient_id, image_id, predicted_class, confidence, severity_level, advice)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (patient_id, image_id, predicted_class,
          confidence, severity_level, advice))
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id


def get_all_predictions():
    """Return all predictions joined with patient names."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ph.*, p.full_name, ui.filename
        FROM prediction_history ph
        LEFT JOIN patients p ON ph.patient_id = p.patient_id
        LEFT JOIN uploaded_images ui ON ph.image_id = ui.image_id
        ORDER BY ph.prediction_time DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_predictions_by_patient(patient_id):
    """Return prediction history for a specific patient."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ph.*, ui.filename
        FROM prediction_history ph
        LEFT JOIN uploaded_images ui ON ph.image_id = ui.image_id
        WHERE ph.patient_id = ?
        ORDER BY ph.prediction_time DESC
    """, (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_prediction_by_id(prediction_id):
    """Return a single prediction record with patient info."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ph.*, p.full_name, p.age, p.gender, p.email, ui.filename
        FROM prediction_history ph
        LEFT JOIN patients p ON ph.patient_id = p.patient_id
        LEFT JOIN uploaded_images ui ON ph.image_id = ui.image_id
        WHERE ph.prediction_id = ?
    """, (prediction_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ===========================================================
# REPORT OPERATIONS
# ===========================================================

def insert_report(patient_id, prediction_id, report_filename, report_path):
    """Save a generated report record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reports (patient_id, prediction_id, report_filename, report_path)
        VALUES (?, ?, ?, ?)
    """, (patient_id, prediction_id, report_filename, report_path))
    conn.commit()
    conn.close()


def get_statistics():
    """Return dashboard statistics."""
    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    cursor.execute("SELECT COUNT(*) AS cnt FROM patients")
    stats["total_patients"] = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) AS cnt FROM prediction_history")
    stats["total_predictions"] = cursor.fetchone()["cnt"]

    cursor.execute("""
        SELECT predicted_class, COUNT(*) AS cnt
        FROM prediction_history
        GROUP BY predicted_class
    """)
    stats["class_distribution"] = {row["predicted_class"]: row["cnt"]
                                    for row in cursor.fetchall()}

    cursor.execute("""
        SELECT AVG(confidence) AS avg_conf FROM prediction_history
    """)
    row = cursor.fetchone()
    stats["avg_confidence"] = round(row["avg_conf"] or 0, 2)

    conn.close()
    return stats


# Run initialization when this module is imported
if not os.path.exists(os.path.dirname(DB_PATH)):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

initialize_database()
