import os
import joblib
import yaml
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class Predictor:
    def __init__(self):
        self.model_path = self.load_config()['models'][0]['store_path']  # use first model's path
        self.pipeline = self.load_model()

    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def load_model(self):
        # Automatically load the only .pkl model in the path
        for file in os.listdir(self.model_path):
            if file.endswith('.pkl'):
                model_file_path = os.path.join(self.model_path, file)
                print(f"Loading model from: {model_file_path}")
                return joblib.load(model_file_path)
        raise FileNotFoundError("No .pkl model found in the model directory.")

    def feature_target_separator(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def evaluate_model(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        return accuracy, class_report, roc_auc
