# TODO: Import numpy, pickle for model loading
# TODO: Import settings for model paths
# TODO: Create async predict_goal_success function accepting goal and financial data
# TODO: Load goal model and LabelEncoder from disk or use defaults
# TODO: Prepare features array from goal data and financial metrics
# TODO: Call model.predict and model.predict_proba
# TODO: Extract success prediction and probability
# TODO: Create fallback rule-based prediction if model unavailable:
#   - Check annual income vs target goal amount
#   - Check investment horizon
#   - Apply financial feasibility rules
# TODO: Return (goal_achievable, probability) tuple

import numpy as np 
import pickle 
import joblib 
import logging 
import pandas as pd 
from typing import Dict,Any,Tuple
import os
import asyncio

logger=logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_goal_model():
    MODEL_DIR = "ml/models/goal"
    goal_MODEL_PATH = os.path.join(MODEL_DIR,"goal_success_model.pkl")
    goal_FEATURE_ENCODER_PATH = os.path.join(MODEL_DIR,"goal_feature_encoder.pkl")
    goal_TARGET_ENCODER_PATH = os.path.join(MODEL_DIR,"goal_target_encoder.pkl")

    try:
        model=joblib.load(goal_MODEL_PATH)
        feature_encoder=joblib.load(goal_FEATURE_ENCODER_PATH)
        target_encoder=joblib.load(goal_TARGET_ENCODER_PATH)
        logger.info("Loaded goal model and its encoders")

        return model,feature_encoder,target_encoder

    except FileNotFoundError as e:
        logger.warning("goal model files not found, using fallback rules instead")
        


async def predict_goal_success(prospect_data:Dict[str,Any]) -> Tuple[str,float]:

    # try:
    #     train_goal_model()
    # except:
    #     logger.info("Training Function not found")

    try:

        model,feature_encoder,target_encoder= load_goal_model() # may have to be commented if clashes with agent logic 
        features=prospect_data.copy()

        if 'investment_experience_level' in features:
            features['investment_experience_level'] = feature_encoder.transform([features['investment_experience_level']])[0]
        
        features_col=["age","annual_income","current_savings","target_goal_amount","investment_horizon_years","number_of_dependents","investment_experience_level"]
        # X=np.array([features[col] for col in features_col]).reshape(1,-1)

        # pred_label_encoded=model.predict(X)[0]
        # pred_proba=model.predict_proba(X).max()

        X = pd.DataFrame([features], columns=features_col)
        pred_label_encoded = model.predict(X)[0]
        pred_proba = model.predict_proba(X).max()

        goal_success=target_encoder.inverse_transform([pred_label_encoded])[0]

        return goal_success,float(pred_proba)
    except Exception as e:
        logger.warning(f"Error running goal model: {e}, using fallback rules instead.")
        return rule_based_scoring_goal(prospect_data)

def rule_based_scoring_goal(prospect_data:Dict[str,Any]) -> Tuple[str,float]:
    savings=prospect_data.get("current_savings",0)
    horizon=prospect_data.get("investment_horizon_years",0)
    goal_amount = prospect_data.get("target_goal_amount")

    savings_ratio = savings/goal_amount 

    if savings_ratio > 0.4 and horizon >= 10:
        return "Likely"
    elif savings_ratio < 0.2 and horizon <5:
        return "Unlikely"
    else:
        return "Moderate"

        

if __name__ == "__main__":
    prospect={
        "age":35,
        "annual_income":738477,
        "current_savings":17000,
        "investment_horizon_years": 1,
        "target_goal_amount":340990,
        "number_of_dependents": 1,
        "investment_experience_level": "Beginner"
    }
    result = asyncio.run(predict_goal_success(prospect))

    if isinstance(result,tuple):
        goal_success,confidence=result
        logger.info(f"Predicted goal Level: {goal_success},Confidence:{confidence:.2f}")
    else:
        goal_success= result
        logger.info(f"Predicted goal Level: {goal_success}") 
