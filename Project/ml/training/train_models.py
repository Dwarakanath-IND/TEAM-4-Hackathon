# TODO: Import pandas, numpy for data handling
# TODO: Import scikit-learn models (RandomForestClassifier)
# TODO: Import LabelEncoder and train_test_split
# TODO: Import pickle for model persistence
# TODO: Create train_risk_model function:
#   - Load training data from CSV
#   - Prepare features and target variable
#   - Split data into train/test sets
#   - Train RandomForest classifier
#   - Evaluate model performance
#   - Save model and encoders to disk
# TODO: Create train_goal_model function:
#   - Load training data
#   - Prepare features for goal success prediction
#   - Train classifier
#   - Evaluate performance
#   - Save model and encoders
# TODO: Create train_models function orchestrating both training pipelines
# TODO: Add progress reporting during training
# TODO: Handle missing model data with informative errors

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import pickle 
import os
import logging
import joblib 

logging.basicConfig(level=logging.INFO)
logger= logging.getLogger(__name__)

def train_risk_model():
    try: 
        logger.info("Loading risk profile training data")
        risk_df=pd.read_csv("ml/data/risk_profile_training_dataset.csv") 
        if risk_df.empty:
            raise ValueError("Risk profile dataset is empty!")

        
        target_col="risk_profile"

        # feature engineering
        # risk_df["feature1"]=risk_df.apply( lambda row:row["annual_income"]/ row["number_of_dependents"]
        #                                     if row['number_of_dependents']>0 else row["annual_income"],axis=1)


        
        label_encoder = LabelEncoder()
        risk_df["investment_experience_level"]=label_encoder.fit_transform(risk_df["investment_experience_level"])
        feature_cols=["age","annual_income","current_savings","investment_horizon_years","number_of_dependents","investment_experience_level"]
        
        X=risk_df[feature_cols]
        y=risk_df[target_col]

        y_label_encoder = LabelEncoder()
        y_encoded = y_label_encoder.fit_transform(y)

        # checking class distribution 
        # for i,label in enumerate(y_label_encoder.classes_):
        #     count=sum(y_encoded == i)
        #     logger.info(f" {label}:{count}({count/len(y_encoded)*100:.1f}%)")


        X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size=0.3, random_state=42)

        #=========================================================================================================hyperparameter tuning

        # from sklearn.model_selection import GridSearchCV
        # from sklearn.model_selection import RandomizedSearchCV 
        # from tqdm.auto import tqdm

        # param_grid={
        #     'n_estimators':[100,200],
        #     'max_depth': [10,15,None],
        #     'min_samples_split':[2,5],
        #     'min_samples_leaf':[1,2],
        #     'max_features':['sqrt','log2'],
        #     'criterion': ['gini','entropy']
        # }

        # logger.info("Starting grid search...")

        # rf=RandomForestClassifier(random_state=42,class_weight='balanced')
        # grid=GridSearchCV(rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=0,scoring="accuracy")
        
        # def tqdm_gridsearch(grid,X,y):
        #     n_candidates * grid.cv 
        #     with tqdm(total=total,desc="Grid Search Progress",position=0) as pbar:
        #         def _fit_and_score_progress(*args,**kwargs):
        #             pbar.update(1)
        #             return _fit_and_score(*args,**kwargs)
        #         original= val._fit_and_score 
        #         val._fit_and_score = _fit_and_score_progress
        #         try: 
        #             grid.fit(X,y)
        #         finally: 
        #             val._fit_and_score = original 
        #     return grid 

        # grid_search = tqdm_gridsearch(grid,X_train,y_train)
        # best_model=grid_search.best_estimator_
        # #print(best_model)

        # print("\n Best Parameters:", grid_search.best_params_)
        # print("Best CV Score:", grid_search.best_score_)

        #============================================================================================tuning done 

        logger.info('Training RandomForest Model for risk assessment...')
        model = RandomForestClassifier(n_estimators= 300, random_state=42,class_weight="balanced",criterion='entropy',
                                        max_depth=15, min_samples_split=5,min_samples_leaf=4, max_features='sqrt',bootstrap=True,max_samples=0.8)
        model.fit(X_train,y_train)

        y_pred_1 = model.predict(X_train)
        acc1 = accuracy_score(y_train, y_pred_1)
        logger.info (f"Risk Model Train Accuracy: {acc1:.2f}")

        y_pred = model.predict(X_test)
        acc= accuracy_score(y_test, y_pred)
        logger.info (f"Risk Model Test Accuracy: {acc:.2f}")
        # logger.info("\n"+ classification_report(y_test,y_pred,target_names=y_label_encoder.classes_))

        # Training a Gradient Boosting Classifier
        gb_model = GradientBoostingClassifier(random_state=1, n_estimators=500, learning_rate=0.1)
        gb_model.fit(X_train, y_train)

        # Making predictions
        y_pred_gb = gb_model.predict(X_test)
        # Evaluating the model
        accuracy_gb = accuracy_score(y_test, y_pred_gb)
        logger.info (f"Risk Model Test Accuracy (Gradient Boosting): {accuracy_gb:.2f}")
        logger.info("\n Classification Report (Gradient Boosting): \n\n" + classification_report(y_test,y_pred_gb,target_names=y_label_encoder.classes_))


        ## saving the model and encoders
        model_path="ml/models/risk/risk_model.pkl"
        joblib.dump(gb_model,model_path)
        logger.info("Risk model saved successfully.")

        joblib.dump(y_label_encoder,"ml/models/risk/risk_target_encoder.pkl")
        joblib.dump(label_encoder,"ml/models/risk/risk_feature_encoder.pkl")
        logger.info("Risk Profile => Encoders saved successfully")

        feature_importances = model.feature_importances_
#         indices = np.argsort(feature_importances)[::-1]

        print(sorted(zip(model.feature_importances_, X.columns), reverse=True))

        plt.figure(figsize=(15, 6))
        plt.bar(range(X.shape[1]), feature_importances, label = "Risk Model Features")
        plt.xticks(range(X.shape[1]), feature_cols, rotation=0, fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(loc="upper right", fontsize=12)
        plt.title('Feature Importances',fontsize=20)
        plt.xlabel('Feature Index', fontsize=15,labelpad= 15)
        plt.ylabel('Importance', fontsize=15, labelpad= 15)
        plt.show()

    except FileNotFoundError:
        logger.error("Risk profile training data not found.")
    except Exception as e:
        logger.error(f"Error Training risk model: {e}")

def train_goal_model():
    try:
        logger.info("Loading goal success training data...")
        goal_df=pd.read_csv("ml/data/goal_success_training_dataset.csv")
        if goal_df.empty:
            raise ValueError("Goal success dataset is empty!")

        target_col="goal_success"

        label_encoder = LabelEncoder()
        goal_df["investment_experience_level"]=label_encoder.fit_transform(goal_df["investment_experience_level"])
        feature_cols=["age","annual_income","current_savings","target_goal_amount","investment_horizon_years","number_of_dependents","investment_experience_level"]
        
        X=goal_df[feature_cols]
        y=goal_df[target_col]

        y_label_encoder = LabelEncoder()
        y_encoded = y_label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators= 100, random_state=42,class_weight="balanced",criterion='entropy',
                                        max_depth=15, min_samples_split=5,min_samples_leaf=4, max_features='sqrt',bootstrap=True,max_samples=0.8)
        model.fit(X_train,y_train)

        y_pred_1 = model.predict(X_train)
        acc1 = accuracy_score(y_train, y_pred_1)
        logger.info (f"Goal Success Model Train Accuracy: {acc1:.2f}")

        y_pred = model.predict(X_test)
        acc= accuracy_score(y_test, y_pred)
        logger.info (f"Goal Success Model Test Accuracy: {acc:.2f}")
        # logger.info("\n"+ classification_report(y_test,y_pred,target_names=y_label_encoder.classes_))

       # Training a Gradient Boosting Classifier
        gb_model = GradientBoostingClassifier(random_state=1, n_estimators=500, learning_rate=0.1)
        gb_model.fit(X_train, y_train)

        # Making predictions
        y_pred_gb = gb_model.predict(X_test)
        # Evaluating the model
        accuracy_gb = accuracy_score(y_test, y_pred_gb)
        logger.info (f"Goal Success Model Test Accuracy (Gradient Boosting): {accuracy_gb:.2f}")
        logger.info("\n Classification Report (Gradient Boosting): \n\n" + classification_report(y_test,y_pred_gb,target_names=y_label_encoder.classes_))
        ## saving the model

        model_path="ml/models/goal/goal_success_model.pkl"
        joblib.dump(gb_model,model_path)
        logger.info("Goal Success model saved successfully.")

        joblib.dump(y_label_encoder,"ml/models/goal/goal_target_encoder.pkl")
        joblib.dump(label_encoder,"ml/models/goal/goal_feature_encoder.pkl")
        logger.info("Goal Success => Target Encoder saved successfully")

        # Plot feature importances
        feature_importances = model.feature_importances_
#         indices = np.argsort(feature_importances)[::-1]

#        print(zip(X.columns, model.feature_importances_))
        print(sorted(zip(model.feature_importances_, X.columns), reverse=True))

        
        plt.figure(figsize=(15, 6))
        plt.bar(range(X.shape[1]), feature_importances, label ="Goal Success Model Features")
        plt.xticks(range(X.shape[1]), feature_cols, rotation=0, fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(loc="upper right", fontsize=12)
        plt.title('Feature Importances',fontsize=20)
        plt.xlabel('Feature Index', fontsize=15,labelpad= 15)
        plt.ylabel('Importance', fontsize=15, labelpad= 15)
        plt.show()

    except FileNotFoundError:
        logger.error("Goal success training data not found.")
    except Exception as e:
        logger.error(f"Error Training goal success model: {e}")


        
if __name__ == "__main__":
    train_risk_model()
    train_goal_model()
