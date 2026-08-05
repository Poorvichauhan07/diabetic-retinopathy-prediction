"""
utils/demo_data.py
Seed the database with sample patient and prediction data for demonstration.
Run: python utils/demo_data.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import database as db

DEMO_PATIENTS = [
    ("Rajesh Kumar",    52, "Male",   "rajesh.kumar@demo.com",   "9876543210", "New Delhi, India"),
    ("Priya Sharma",    45, "Female", "priya.sharma@demo.com",   "9812345678", "Mumbai, India"),
    ("Amit Singh",      61, "Male",   "amit.singh@demo.com",     "9898765432", "Bangalore, India"),
    ("Sunita Verma",    38, "Female", "sunita.verma@demo.com",   "9845123456", "Chennai, India"),
    ("Mohammed Ali",    57, "Male",   "mohammed.ali@demo.com",   "9867890123", "Hyderabad, India"),
    ("Deepa Nair",      49, "Female", "deepa.nair@demo.com",     "9823456789", "Kochi, India"),
    ("Vikram Patel",    66, "Male",   "vikram.patel@demo.com",   "9812398765", "Ahmedabad, India"),
    ("Meera Iyer",      42, "Female", "meera.iyer@demo.com",     "9876512345", "Pune, India"),
]

DEMO_PREDICTIONS = [
    (1, 1, "No DR",           92.4, "No Diabetic Retinopathy",
     "Annual screening recommended. Maintain good glycemic control."),
    (2, 2, "Mild",            87.1, "Mild Non-Proliferative DR",
     "Monitor every 9 months. Improve blood sugar management."),
    (3, 3, "Moderate",        78.5, "Moderate Non-Proliferative DR",
     "Consult ophthalmologist within 3-6 months."),
    (4, 4, "Severe",          85.2, "Severe Non-Proliferative DR",
     "Urgent referral within 1 month. High risk of vision loss."),
    (5, 5, "Proliferative DR",91.7, "Proliferative DR (Vision Threatening)",
     "IMMEDIATE specialist referral. Risk of blindness."),
    (6, 6, "Mild",            83.3, "Mild Non-Proliferative DR",
     "Monitor every 9 months. Review diabetes management plan."),
    (7, 7, "No DR",           95.1, "No Diabetic Retinopathy",
     "No DR detected. Continue annual monitoring."),
    (8, 8, "Moderate",        76.8, "Moderate Non-Proliferative DR",
     "Consult an ophthalmologist soon. Strict glycemic control required."),
]


def seed():
    print("\n[Demo] Seeding database with sample data...")

    # Insert patients and collect IDs
    patient_ids = []
    for p in DEMO_PATIENTS:
        pid = db.insert_patient(*p)
        patient_ids.append(pid)
        print(f"  Patient: {p[0]} → ID {pid}")

    # Insert fake image records
    image_ids = []
    for i, pid in enumerate(patient_ids):
        fname = f"demo_{i+1:02d}.png"
        fpath = os.path.join("uploads", fname)
        iid   = db.insert_image(pid, fname, fpath)
        image_ids.append(iid)

    # Insert predictions
    for i, pred in enumerate(DEMO_PREDICTIONS):
        patient_id, image_idx, cls, conf, sev, advice = pred
        real_pid = patient_ids[patient_id - 1] if patient_id - 1 < len(patient_ids) else patient_ids[0]
        real_iid = image_ids[image_idx - 1]    if image_idx - 1  < len(image_ids)  else image_ids[0]
        pred_id = db.insert_prediction(real_pid, real_iid, cls, conf, sev, advice)
        print(f"  Prediction #{pred_id}: {cls} ({conf}%) for patient {real_pid}")

    print("\n[Demo] Seeding complete! Run 'python main.py' and visit http://localhost:5000\n")


if __name__ == "__main__":
    seed()
