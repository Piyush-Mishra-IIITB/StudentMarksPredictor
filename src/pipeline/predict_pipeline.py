import pandas as pd
import os
import pickle

class PredictPipeline:
    def __init__(self):
        # Paths to artifacts
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

        # Check if files exist
        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Model or preprocessor not found. Make sure these files exist:\n{self.model_path}\n{self.preprocessor_path}"
            )

        # Load model and preprocessor
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

    def predict(self, features: pd.DataFrame):
        # Ensure input columns match preprocessor
        missing_cols = set(self.preprocessor.feature_names_in_) - set(features.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input: {missing_cols}")

        # Transform and predict
        input_scaled = self.preprocessor.transform(features)
        preds = self.model.predict(input_scaled)
        return preds


class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch,
                 test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        return pd.DataFrame({
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score],
        })
