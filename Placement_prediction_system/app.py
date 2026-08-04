"""
SmartPlacement — Flask UI
--------------------------
This file wires up the pages and a placeholder prediction function.
Everything under "DUMMY DATA" and "PLACEHOLDER MODEL LOGIC" is here only
so the UI is runnable end-to-end. Swap those two sections for your real
dataset / trained model (e.g. joblib.load("model.pkl")) — the templates
and routes don't need to change as long as you keep the same variable
names being passed to render_template().
"""

import random
from datetime import datetime
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# DUMMY DATA  (replace with your real dataset / DB query results)
# ---------------------------------------------------------------------------

TOTAL_STUDENTS = 50000
PLACED_STUDENTS = 28732
NOT_PLACED_STUDENTS = TOTAL_STUDENTS - PLACED_STUDENTS
AVG_SALARY = 7.25

SAMPLE_NAMES = [
    "Rohan Verma", "Priya Sharma", "Aditya Singh", "Neha Gupta", "Karan Patel",
    "Ananya Rao", "Ishaan Mehta", "Divya Nair", "Arjun Reddy", "Sneha Iyer",
]

def _generate_students(n=40):
    random.seed(7)
    rows = []
    for i in range(n):
        placed = random.random() < 0.574
        rows.append({
            "name": SAMPLE_NAMES[i % len(SAMPLE_NAMES)],
            "cgpa": round(random.uniform(6.0, 9.8), 2),
            "internships": random.randint(0, 4),
            "aptitude": random.randint(40, 98),
            "placed": placed,
            "salary": round(random.uniform(3.5, 14.0), 2) if placed else None,
        })
    return rows

STUDENTS = _generate_students()

SALARY_LABELS = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]
SALARY_VALUES = [4, 22, 41, 18, 10, 5]  # % of placed students in each band

CGPA_LABELS = ["6-7", "7-8", "8-9", "9-10"]
CGPA_VALUES = [32, 51, 74, 91]  # placement rate % per CGPA band

BRANCH_LABELS = ["CSE", "ECE", "MECH", "Others"]
BRANCH_VALUES = [47, 23, 14, 16]  # % share

AVG_CGPA = 7.85
AVG_APTITUDE = 72.31

TREND_LABELS = ["2021", "2022", "2023", "2024", "2025", "2026"]
TREND_VALUES = [49.2, 51.8, 53.6, 55.1, 56.4, 57.46]  # placement rate % per batch year


# ---------------------------------------------------------------------------
# PLACEHOLDER MODEL LOGIC  (replace with model.predict(...))
# ---------------------------------------------------------------------------

def predict_placement(cgpa, soft_skills, internships, backlogs, projects,
                       extra_curricular, aptitude, communication):
    """Toy scoring function standing in for a trained classifier + regressor.
    Replace the body of this function with your actual model inference."""
    score = (
        cgpa * 8
        + soft_skills * 2.5
        + internships * 6
        + projects * 3
        - backlogs * 12
        + extra_curricular * 1.2
        + aptitude * 0.35
        + communication * 2.5
    )
    prob_placed = max(2, min(98, round(score / 1.8)))
    placed = prob_placed >= 50

    base_salary = 3.0 + (cgpa - 6) * 1.1 + internships * 0.6 + projects * 0.3 + aptitude * 0.02
    salary = round(max(3.0, base_salary), 2) if placed else round(max(2.5, base_salary * 0.55), 2)

    return {
        "placed": placed,
        "prob_placed": prob_placed,
        "prob_not_placed": 100 - prob_placed,
        "salary": salary,
    }


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.context_processor
def inject_year():
    return {"current_year": datetime.now().year}


@app.route("/")
def dashboard():
    placed_pct = round(PLACED_STUDENTS / TOTAL_STUDENTS * 100, 2)
    return render_template(
        "dashboard.html",
        active="dashboard",
        total_students=TOTAL_STUDENTS,
        placed_students=PLACED_STUDENTS,
        not_placed_students=NOT_PLACED_STUDENTS,
        avg_salary=AVG_SALARY,
        placed_pct=placed_pct,
        not_placed_pct=round(100 - placed_pct, 2),
        salary_labels=SALARY_LABELS,
        salary_values=SALARY_VALUES,
        trend_labels=TREND_LABELS,
        trend_values=TREND_VALUES,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form_values = None
    if request.method == "POST":
        form_values = {
            "cgpa": request.form.get("cgpa", type=float),
            "soft_skills": request.form.get("soft_skills", type=float),
            "internships": request.form.get("internships", type=int),
            "backlogs": request.form.get("backlogs", type=int),
            "projects": request.form.get("projects", type=int),
            "extra_curricular": request.form.get("extra_curricular", type=float),
            "aptitude": request.form.get("aptitude", type=float),
            "communication": request.form.get("communication", type=float),
        }
        result = predict_placement(**form_values)

    return render_template("predict.html", active="predict", result=result, form_values=form_values)


@app.route("/students")
def students():
    return render_template("students.html", active="students", students=STUDENTS)


@app.route("/analytics")
def analytics():
    placed_pct = round(PLACED_STUDENTS / TOTAL_STUDENTS * 100, 2)
    return render_template(
        "analytics.html",
        active="analytics",
        avg_cgpa=AVG_CGPA,
        placed_pct=placed_pct,
        avg_aptitude=AVG_APTITUDE,
        avg_salary=AVG_SALARY,
        cgpa_labels=CGPA_LABELS,
        cgpa_values=CGPA_VALUES,
        branch_labels=BRANCH_LABELS,
        branch_values=BRANCH_VALUES,
    )


@app.route("/about")
def about():
    return render_template("about.html", active="about")


if __name__ == "__main__":
    app.run(debug=True)
