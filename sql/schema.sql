-- ============================================================
-- sql/schema.sql
-- Diabetic Retinopathy Prediction System — Database Schema
-- Compatible with SQLite3
-- ============================================================

-- Drop tables if they exist (for fresh setup)
DROP TABLE IF EXISTS reports;
DROP TABLE IF EXISTS prediction_history;
DROP TABLE IF EXISTS uploaded_images;
DROP TABLE IF EXISTS patients;

-- ──────────────────────────────────────────────────────────────
-- TABLE: patients
-- Stores patient demographic information
-- ──────────────────────────────────────────────────────────────
CREATE TABLE patients (
    patient_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name    TEXT    NOT NULL,
    age          INTEGER CHECK (age >= 0 AND age <= 150),
    gender       TEXT    CHECK (gender IN ('Male', 'Female', 'Other') OR gender IS NULL),
    email        TEXT    UNIQUE,
    phone        TEXT,
    address      TEXT,
    created_at   TEXT    DEFAULT (datetime('now')),
    updated_at   TEXT    DEFAULT (datetime('now'))
);

-- Index on name for fast search
CREATE INDEX idx_patients_name  ON patients(full_name);
CREATE INDEX idx_patients_email ON patients(email);

-- ──────────────────────────────────────────────────────────────
-- TABLE: uploaded_images
-- Tracks every retinal image uploaded by users
-- ──────────────────────────────────────────────────────────────
CREATE TABLE uploaded_images (
    image_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   INTEGER NOT NULL,
    filename     TEXT    NOT NULL,
    filepath     TEXT    NOT NULL,
    upload_time  TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_images_patient ON uploaded_images(patient_id);

-- ──────────────────────────────────────────────────────────────
-- TABLE: prediction_history
-- Stores each AI prediction result
-- ──────────────────────────────────────────────────────────────
CREATE TABLE prediction_history (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    image_id        INTEGER,
    predicted_class TEXT    NOT NULL
                    CHECK (predicted_class IN
                           ('No DR','Mild','Moderate','Severe','Proliferative DR')),
    confidence      REAL    NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
    severity_level  TEXT,
    advice          TEXT,
    prediction_time TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        ON DELETE CASCADE,
    FOREIGN KEY (image_id)   REFERENCES uploaded_images(image_id)
        ON DELETE SET NULL
);

CREATE INDEX idx_pred_patient ON prediction_history(patient_id);
CREATE INDEX idx_pred_class   ON prediction_history(predicted_class);
CREATE INDEX idx_pred_time    ON prediction_history(prediction_time);

-- ──────────────────────────────────────────────────────────────
-- TABLE: reports
-- Tracks generated PDF reports
-- ──────────────────────────────────────────────────────────────
CREATE TABLE reports (
    report_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id        INTEGER NOT NULL,
    prediction_id     INTEGER,
    report_filename   TEXT    NOT NULL,
    report_path       TEXT    NOT NULL,
    generated_at      TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (patient_id)    REFERENCES patients(patient_id)
        ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES prediction_history(prediction_id)
        ON DELETE SET NULL
);

-- ──────────────────────────────────────────────────────────────
-- SAMPLE / DEMO DATA (for testing — can be removed in production)
-- ──────────────────────────────────────────────────────────────
INSERT INTO patients (full_name, age, gender, email, phone, address)
VALUES
  ('Rajesh Kumar',    52, 'Male',   'rajesh.kumar@example.com',   '9876543210', 'Delhi, India'),
  ('Priya Sharma',    45, 'Female', 'priya.sharma@example.com',   '9812345678', 'Mumbai, India'),
  ('Amit Singh',      61, 'Male',   'amit.singh@example.com',     '9898765432', 'Bangalore, India'),
  ('Sunita Verma',    38, 'Female', 'sunita.verma@example.com',   '9845123456', 'Chennai, India'),
  ('Mohammed Ali',    57, 'Male',   'mohammed.ali@example.com',   '9867890123', 'Hyderabad, India'),
  ('Deepa Nair',      49, 'Female', 'deepa.nair@example.com',     '9823456789', 'Kochi, India');

INSERT INTO uploaded_images (patient_id, filename, filepath)
VALUES
  (1, 'demo_01.png', 'uploads/demo_01.png'),
  (2, 'demo_02.png', 'uploads/demo_02.png'),
  (3, 'demo_03.png', 'uploads/demo_03.png'),
  (4, 'demo_04.png', 'uploads/demo_04.png'),
  (5, 'demo_05.png', 'uploads/demo_05.png'),
  (6, 'demo_06.png', 'uploads/demo_06.png');

INSERT INTO prediction_history
  (patient_id, image_id, predicted_class, confidence, severity_level, advice)
VALUES
  (1, 1, 'No DR',           92.4, 'No Diabetic Retinopathy',                   'Annual screening recommended. Maintain good glycemic control.'),
  (2, 2, 'Mild',            87.1, 'Mild Non-Proliferative DR',                  'Monitor every 9 months. Improve blood sugar management.'),
  (3, 3, 'Moderate',        78.5, 'Moderate Non-Proliferative DR',              'Consult ophthalmologist within 3-6 months.'),
  (4, 4, 'Severe',          85.2, 'Severe Non-Proliferative DR',                'Urgent referral within 1 month. High risk of vision loss.'),
  (5, 5, 'Proliferative DR',91.7, 'Proliferative DR (Vision Threatening)',      'IMMEDIATE specialist referral. Risk of blindness.'),
  (6, 6, 'Mild',            83.3, 'Mild Non-Proliferative DR',                  'Monitor every 9 months. Review diabetes management plan.');

-- ──────────────────────────────────────────────────────────────
-- USEFUL QUERIES (for reference / reporting)
-- ──────────────────────────────────────────────────────────────

-- Count predictions per DR class:
-- SELECT predicted_class, COUNT(*) AS total
-- FROM prediction_history GROUP BY predicted_class ORDER BY total DESC;

-- Get full patient history:
-- SELECT p.full_name, ph.predicted_class, ph.confidence, ph.prediction_time
-- FROM prediction_history ph
-- JOIN patients p ON ph.patient_id = p.patient_id
-- ORDER BY ph.prediction_time DESC;

-- Average confidence per class:
-- SELECT predicted_class, ROUND(AVG(confidence), 2) AS avg_confidence
-- FROM prediction_history GROUP BY predicted_class;

-- Patients with severe or proliferative DR:
-- SELECT p.full_name, p.phone, ph.predicted_class, ph.prediction_time
-- FROM prediction_history ph
-- JOIN patients p ON ph.patient_id = p.patient_id
-- WHERE ph.predicted_class IN ('Severe', 'Proliferative DR');
