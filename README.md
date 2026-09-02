# Disease Prediction Project

This project is a beginner-friendly disease prediction system built with Python and machine learning. It uses a synthetic symptom dataset to train a Random Forest classifier that predicts likely diseases from selected symptoms.

## Features

- Symptom-based disease prediction
- Easy-to-run Streamlit web app
- Synthetic dataset for demonstration
- Beginner-friendly ML workflow

## Project Structure

- `app.py`: Streamlit user interface
- `disease_prediction.py`: data generation, model training, and prediction logic
- `requirements.txt`: project dependencies

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Example

If a user selects symptoms like `fever`, `cough`, and `loss_of_smell`, the model may predict `covid_19`.

## Notes

This is a demonstration project using a synthetic dataset. For production deployment, use a real medical dataset and consult healthcare professionals.
