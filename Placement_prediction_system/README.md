# SmartPlacement — UI

A Flask front end for a placement prediction system: Dashboard, Predict,
Students, Analytics, and About pages, with a light/dark theme toggle.

## Run it

```
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000

## Project structure

```
app.py                  Routes + placeholder data/logic
templates/
  base.html              Shared shell: sidebar, topbar, theme toggle
  dashboard.html
  predict.html
  students.html
  analytics.html
  about.html
static/
  css/style.css          All design tokens + component styles
  js/theme.js             Theme toggle, count-up numbers, mobile nav
  js/charts.js             Chart.js setup for donut/bar/pie charts
```

## Wiring up your real ML model

Everything you need to replace lives in `app.py`, in two clearly marked sections:

1. **`DUMMY DATA`** — swap `TOTAL_STUDENTS`, `PLACED_STUDENTS`, `STUDENTS`,
   `SALARY_VALUES`, `CGPA_VALUES`, `BRANCH_VALUES`, etc. with values pulled
   from your real dataset (CSV, DB query, whatever you're using).

2. **`predict_placement()`** — this is currently a toy weighted-score
   function. Replace its body with your actual model inference, e.g.:

   ```python
   import joblib
   classifier = joblib.load("models/placement_classifier.pkl")
   regressor = joblib.load("models/salary_regressor.pkl")

   def predict_placement(cgpa, soft_skills, internships, backlogs, projects,
                          extra_curricular, aptitude, communication):
       X = [[cgpa, soft_skills, internships, backlogs, projects,
             extra_curricular, aptitude, communication]]
       placed = bool(classifier.predict(X)[0])
       prob_placed = round(classifier.predict_proba(X)[0][1] * 100)
       salary = round(float(regressor.predict(X)[0]), 2)
       return {
           "placed": placed,
           "prob_placed": prob_placed,
           "prob_not_placed": 100 - prob_placed,
           "salary": salary,
       }
   ```

   The route and template already expect exactly this return shape, so no
   other changes are needed.

## Notes

- Theme choice is stored in the browser (`localStorage`), so it's remembered
  per visitor across sessions.
- The "seal" mark in the sidebar and the stamp on the prediction result are
  plain inline SVG — no icon library dependency.
- All colors are CSS variables in `static/css/style.css`, so dark mode is a
  single attribute swap (`data-theme="dark"` on `<html>`), not a duplicate
  stylesheet.
