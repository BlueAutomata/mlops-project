import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize FastAPI
app = FastAPI()

# Define the request body format for predictions
class PredictionFeatures(BaseModel):
    experience_level_encoded: float
    company_size_encoded: float
    employment_type_FL: int
    employment_type_FT: int
    employment_type_PT: int
    job_title_AI_Scientist: int
    job_title_Analytics_Engineer: int
    job_title_Applied_Data_Scientist: int
    job_title_Applied_Machine_Learning_Scientist: int
    job_title_BI_Data_Analyst: int
    job_title_Big_Data_Architect: int
    job_title_Big_Data_Engineer: int
    job_title_Business_Data_Analyst: int
    job_title_Cloud_Data_Engineer: int
    job_title_Computer_Vision_Engineer: int
    job_title_Computer_Vision_Software_Engineer: int
    job_title_Data_Analyst: int
    job_title_Data_Analytics_Engineer: int
    job_title_Data_Analytics_Lead: int
    job_title_Data_Analytics_Manager: int
    job_title_Data_Architect: int
    job_title_Data_Engineer: int
    job_title_Data_Engineering_Manager: int
    job_title_Data_Science_Consultant: int
    job_title_Data_Science_Engineer: int
    job_title_Data_Science_Manager: int
    job_title_Data_Scientist: int
    job_title_Data_Specialist: int
    job_title_Director_of_Data_Engineering: int
    job_title_Director_of_Data_Science: int
    job_title_ETL_Developer: int
    job_title_Finance_Data_Analyst: int
    job_title_Financial_Data_Analyst: int
    job_title_Head_of_Data: int
    job_title_Head_of_Data_Science: int
    job_title_Head_of_Machine_Learning: int
    job_title_Lead_Data_Analyst: int
    job_title_Lead_Data_Engineer: int
    job_title_Lead_Data_Scientist: int
    job_title_Lead_Machine_Learning_Engineer: int
    job_title_ML_Engineer: int
    job_title_Machine_Learning_Developer: int
    job_title_Machine_Learning_Engineer: int
    job_title_Machine_Learning_Infrastructure_Engineer: int
    job_title_Machine_Learning_Manager: int
    job_title_Machine_Learning_Scientist: int
    job_title_Marketing_Data_Analyst: int
    job_title_NLP_Engineer: int
    job_title_Principal_Data_Analyst: int
    job_title_Principal_Data_Engineer: int
    job_title_Principal_Data_Scientist: int
    job_title_Product_Data_Analyst: int
    job_title_Research_Scientist: int
    job_title_Staff_Data_Scientist: int


    

# Global variable to store the loaded model
model = None

# Download the model
def download_model():
    global model
    model = joblib.load('models/knn_regress.pkl')

# Download the model immediately when the script runs
download_model()

# API Root endpoint
@app.get("/")
async def index():
    return {"message": "Welcome to the Data Science Income API. Use the /predict feature to predict your income."}

# Prediction endpoint
@app.post("/predict")
async def predict(features: PredictionFeatures):

    # Create input DataFrame for prediction
    input_data = pd.DataFrame([{
        "experience_level_encoded": features.experience_level_encoded,
        "company_size_encoded": features.company_size_encoded,
        "employment_type_FL": features.employment_type_FL,
        "employment_type_FT": features.employment_type_FT,
        "employment_type_PT": features.employment_type_PT,
        "job_title_AI Scientist": features.job_title_AI_Scientist,
        "job_title_Analytics Engineer": features.job_title_Analytics_Engineer,
        "job_title_Applied Data Scientist": features.job_title_Applied_Data_Scientist,
        "job_title_Applied Machine Learning Scientist": features.job_title_Applied_Machine_Learning_Scientist,
        "job_title_BI Data Analyst": features.job_title_BI_Data_Analyst,
        "job_title_Big Data Architect": features.job_title_Big_Data_Architect,
        "job_title_Big Data Engineer": features.job_title_Big_Data_Engineer,
        "job_title_Business Data Analyst": features.job_title_Business_Data_Analyst,
        "job_title_Cloud Data Engineer": features.job_title_Cloud_Data_Engineer,
        "job_title_Computer Vision Engineer": features.job_title_Computer_Vision_Engineer,
        "job_title_Computer Vision Software Engineer": features.job_title_Computer_Vision_Software_Engineer,
        "job_title_Data Analyst": features.job_title_Data_Analyst,
        "job_title_Data Analytics Engineer": features.job_title_Data_Analytics_Engineer,
        "job_title_Data Analytics Lead": features.job_title_Data_Analytics_Lead,
        "job_title_Data Analytics Manager": features.job_title_Data_Analytics_Manager,
        "job_title_Data Architect": features.job_title_Data_Architect,
        "job_title_Data Engineer": features.job_title_Data_Engineer,
        "job_title_Data Engineering Manager": features.job_title_Data_Engineering_Manager,
        "job_title_Data Science Consultant": features.job_title_Data_Science_Consultant,
        "job_title_Data Science Engineer": features.job_title_Data_Science_Engineer,
        "job_title_Data Science Manager": features.job_title_Data_Science_Manager,
        "job_title_Data Scientist": features.job_title_Data_Scientist,
        "job_title_Data Specialist": features.job_title_Data_Specialist,
        "job_title_Director of Data Engineering": features.job_title_Director_of_Data_Engineering,
        "job_title_Director of Data Science": features.job_title_Director_of_Data_Science,
        "job_title_ETL Developer": features.job_title_ETL_Developer,
        "job_title_Finance Data Analyst": features.job_title_Finance_Data_Analyst,
        "job_title_Financial Data Analyst": features.job_title_Financial_Data_Analyst,
        "job_title_Head of Data": features.job_title_Head_of_Data,
        "job_title_Head of Data Science": features.job_title_Head_of_Data_Science,
        "job_title_Head of Machine Learning": features.job_title_Head_of_Machine_Learning,
        "job_title_Lead Data Analyst": features.job_title_Lead_Data_Analyst,
        "job_title_Lead Data Engineer": features.job_title_Lead_Data_Engineer,
        "job_title_Lead Data Scientist": features.job_title_Lead_Data_Scientist,
        "job_title_Lead Machine Learning Engineer": features.job_title_Lead_Machine_Learning_Engineer,
        "job_title_ML Engineer": features.job_title_ML_Engineer,
        "job_title_Machine Learning Developer": features.job_title_Machine_Learning_Developer,
        "job_title_Machine Learning Engineer": features.job_title_Machine_Learning_Engineer,
        "job_title_Machine Learning Infrastructure Engineer": features.job_title_Machine_Learning_Infrastructure_Engineer,
        "job_title_Machine Learning Manager": features.job_title_Machine_Learning_Manager,
        "job_title_Machine Learning Scientist": features.job_title_Machine_Learning_Scientist,
        "job_title_Marketing Data Analyst": features.job_title_Marketing_Data_Analyst,
        "job_title_NLP Engineer": features.job_title_NLP_Engineer,
        "job_title_Principal Data Analyst": features.job_title_Principal_Data_Analyst,
        "job_title_Principal Data Engineer": features.job_title_Principal_Data_Engineer,
        "job_title_Principal Data Scientist": features.job_title_Principal_Data_Scientist,
        "job_title_Product Data Analyst": features.job_title_Product_Data_Analyst,
        "job_title_Research Scientist": features.job_title_Research_Scientist,
        "job_title_Staff Data Scientist": features.job_title_Staff_Data_Scientist
    }])


    # Predict using the loaded model
    prediction = model.predict(input_data)[0]

    return {
        "Salary (USD)": prediction
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)