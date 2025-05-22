import os
import joblib
import yaml
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_configs = self.config['models']
        self.model_path = self.model_configs[0]['store_path']  # assuming all models share the same store path
        self.pipeline = None
        self.best_model_name = None
        self.best_score = -1

        # Map model names to classes
        self.model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier
        }

    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def feature_target_separator(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def create_preprocessor(self):
        return ColumnTransformer(transformers=[
            ('minmax', MinMaxScaler(), ['AnnualPremium']),
            ('standardize', StandardScaler(), ['Age','RegionID']),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'PastAccident']),
        ])

    def train_model(self, X_train, y_train):
        preprocessor = self.create_preprocessor()
        smote = SMOTE(sampling_strategy=1.0)

        for model_cfg in self.model_configs:
            model_name = model_cfg['name']
            model_params = model_cfg['params']
            model_class = self.model_map[model_name]
            model = model_class(**{k: v for k, v in model_params.items() if v is not None})

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('smote', smote),
                ('model', model)
            ])

            scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            avg_score = np.mean(scores)
            print(f"Model: {model_name}, Cross-Validation Accuracy: {avg_score:.4f}")

            if avg_score > self.best_score:
                self.best_score = avg_score
                self.pipeline = pipeline.fit(X_train, y_train)
                self.best_model_name = model_name

    def save_model(self):
        model_file_path = os.path.join(self.model_path, f'{self.best_model_name}.pkl')
        joblib.dump(self.pipeline, model_file_path)
