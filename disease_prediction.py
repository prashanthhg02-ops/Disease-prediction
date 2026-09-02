import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SYMPTOMS = [
    "fever",
    "cough",
    "fatigue",
    "shortness_of_breath",
    "headache",
    "sore_throat",
    "vomiting",
    "diarrhea",
    "chest_pain",
    "body_ache",
    "loss_of_smell",
    "runny_nose",
    "nausea",
    "wheezing",
    "rash",
    "joint_pain",
]

DISEASE_PROFILES = {
    "flu": ["fever", "cough", "fatigue", "body_ache", "headache", "sore_throat"],
    "common_cold": ["runny_nose", "sore_throat", "cough", "fatigue", "headache"],
    "covid_19": ["fever", "cough", "fatigue", "loss_of_smell", "shortness_of_breath", "body_ache"],
    "pneumonia": ["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue"],
    "asthma": ["wheezing", "shortness_of_breath", "cough", "fatigue"],
    "food_poisoning": ["vomiting", "diarrhea", "nausea", "fever"],
    "migraine": ["headache", "nausea", "vomiting", "fatigue"],
    "dengue": ["fever", "body_ache", "headache", "fatigue", "rash"],
}


def generate_synthetic_data(rows_per_disease=40):
    rng = np.random.default_rng(42)
    data = []

    for disease, symptoms in DISEASE_PROFILES.items():
        for _ in range(rows_per_disease):
            record = {symptom: 0 for symptom in SYMPTOMS}

            for symptom in symptoms:
                record[symptom] = 1

            # add some realistic noise
            for symptom in SYMPTOMS:
                if symptom not in symptoms and rng.random() < 0.08:
                    record[symptom] = 1

            record["Disease"] = disease
            data.append(record)

    return pd.DataFrame(data)


def train_model():
    df = generate_synthetic_data()
    X = df[SYMPTOMS]
    y = df["Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, float(accuracy)


def predict_disease(symptoms, model=None):
    if model is None:
        model, _ = train_model()

    symptoms_set = set(symptoms)
    row = {symptom: 1 if symptom in symptoms_set else 0 for symptom in SYMPTOMS}
    result = model.predict(pd.DataFrame([row]))[0]
    return str(result)


if __name__ == "__main__":
    model, accuracy = train_model()
    print(f"Model accuracy: {accuracy:.2f}")
    sample = ["fever", "cough", "fatigue", "loss_of_smell"]
    print(f"Sample prediction for {sample}: {predict_disease(sample, model)}")
