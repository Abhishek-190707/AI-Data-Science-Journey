# Student Performance Prediction Dashboard

A Streamlit dashboard for the UCI Student Performance Mathematics dataset.
https://ai-data-science-journey-cn4sjr9a4s3ruhxhi6nkj6.streamlit.app/

## Dashboard sections
1. **Student Data Analysis** — dataset overview, grade distribution, study time and absences visualizations.
2. **Model Performance** — trains a Random Forest model, reports MAE/RMSE/R², shows actual-vs-predicted and residual plots, and optionally calculates permutation importance.
3. **Grade Prediction** — accepts student attributes and estimates the final Mathematics grade (`G3`).

## Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `student-mat.csv` — place the UCI Mathematics CSV in this folder, or upload it in the app sidebar

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app expects the original UCI `student-mat.csv` file (semicolon-separated). If the file is not beside `app.py`, upload it using the sidebar file uploader.

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `student-mat.csv` into one GitHub folder.
2. Open Streamlit Community Cloud and connect your GitHub account.
3. Choose the repository, branch, and `app.py` as the main file.
4. Deploy the app.

## Important notes
- The app retrains a Random Forest model using an 80/20 split with `random_state=42`.
- Dashboard metrics are recalculated by the app and may differ from the notebook if its preprocessing or model settings differ.
- Predictions are educational estimates, not official grades or validated decisions about students.
- The dataset is small (395 records); model performance should not be assumed to generalize to other schools or populations.
