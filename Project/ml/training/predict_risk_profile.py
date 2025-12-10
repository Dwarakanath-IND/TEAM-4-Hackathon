# TODO: Import numpy, pickle for model loading

import numpy as np 
import pickle 
import joblib 
import logging 
from typing import Dict,Any,Tuple
import os
import pandas as pd
import asyncio

logger=logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO: Import settings for model paths

def load_risk_model():
    MODEL_DIR = "ml/models/risk"
    RISK_MODEL_PATH = os.path.join(MODEL_DIR,"risk_model.pkl")
    RISK_FEATURE_ENCODER_PATH = os.path.join(MODEL_DIR,"risk_feature_encoder.pkl")
    RISK_TARGET_ENCODER_PATH = os.path.join(MODEL_DIR,"risk_target_encoder.pkl")

    try:
        model=joblib.load(RISK_MODEL_PATH)
        feature_encoder=joblib.load(RISK_FEATURE_ENCODER_PATH)
        target_encoder=joblib.load(RISK_TARGET_ENCODER_PATH)
        logger.info("Loaded Risk model and its encoders")
        return model,feature_encoder,target_encoder
    except FileNotFoundError as e:
        logger.warning("Risk model files not found, using fallback rules instead")
        #return rule_based_scoring(prospect_data)

# TODO: Create async predict_risk_profile function accepting prospect features
# TODO: Load risk model and LabelEncoder from disk or use defaults
# TODO: Prepare features array from prospect data
# TODO: Call model.predict and model.predict_proba
# TODO: Extract risk level and confidence score
# TODO: Create fallback rule-based prediction if model unavailable:
#   - Check age, income, savings, investment experience
#   - Apply investment rules to determine risk level
# TODO: Return (risk_level, confidence_score) tuple

async def predict_risk_profile(prospect_data:Dict[str,Any]) -> Tuple[str,float]:
    # try:
    #     train_risk_model()
    # except:
    #     logger.info("Training Function not found")
    try:

        model,feature_encoder,target_encoder= load_risk_model() # may have to be commented if clashes with agent logic 
        #logger.info("Loaded Risk model and its encoders")

        features=prospect_data.copy()

        if 'investment_experience_level' in features:
            features['investment_experience_level'] = feature_encoder.transform([features['investment_experience_level']])[0]
        
        features_col=["age","annual_income","current_savings","investment_horizon_years","number_of_dependents","investment_experience_level"]
        # X=np.array([features[col] for col in features_col]).reshape(1,-1)

        # pred_label_encoded=model.predict(X)[0]
        # pred_proba=model.predict_proba(X).max()
        # print(pred_label_encoded)

        X = pd.DataFrame([features], columns=features_col)

        pred_label_encoded = model.predict(X)[0]
        pred_proba = model.predict_proba(X).max()

        risk_level=target_encoder.inverse_transform([pred_label_encoded])[0]

        return risk_level,float(pred_proba)
    except Exception as e:
        logger.warning(f"Error loading risk model: {e}, using fallback rules instead.")
        return rule_based_scoring(prospect_data)
        
def rule_based_scoring(prospect_data:Dict[str,Any]) -> Tuple[str,float]:
    if isinstance(prospect_data,dict):
        prospects=[prospect_data]
    else:
        prospects=prospect_data

    results=[]
    for prospect in prospects:
        score=0
    
        age=prospect.get("age")
        if age<=25:
            score+=2
        elif age>25 and age<60:
            score+=1
        else:
            score+=0

        income=prospect.get("annual_income")
        if income>=500000:
            score+=1
        elif income>=100000:
            score+=2
        else:
            score+=0 

        hor=prospect.get("investment_horizon_years")
        if hor>=10:
            score+=2 
        elif hor>=5:
            score+=1
        else:
            score+=0 

        exp=prospect.get("investment_experience_level")
        if exp=="Advanced":
            score+=2
        elif exp=="Moderate":
            score+=1
        else:
            score+=0 

        dep=prospect.get("number_of_dependents")
        if dep>=2:
            score+=1 
        print("score=>",score)


        #final score
        if score>=6:
            return "high",score
        elif score>=3:
            return "moderate",score
        else:
            return "low",score



if __name__ == "__main__":
    prospect={
        "age":35,
        "annual_income":485495,
        "current_savings":17298,
        "investment_horizon_years": 8,
        "number_of_dependents": 3,
        "investment_experience_level": "Beginner"
    }

    risk_level,confidence=asyncio.run(predict_risk_profile(prospect))
    logger.info(f"Predicted Risk Level: {risk_level},Confidence:{confidence:.2f}")
