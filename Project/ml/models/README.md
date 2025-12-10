# Models Directory

This directory should contain the pre-trained machine learning models for the RM-AgenticAI system.

## Expected Files

- `risk_profile_model.pkl` - Trained risk assessment model
- `goal_success_model.pkl` - Goal success prediction model
- `label_encoders.pkl` - Label encoders for risk model features
- `goal_success_label_encoders.pkl` - Label encoders for goal model features

## Note

The ML models are not included in this repository due to their size. The system includes fallback mechanisms:

1. **Rule-based risk assessment** when ML models are not available
2. **Heuristic goal prediction** as backup
3. **Graceful degradation** with clear logging

## Training Your Own Models

To train your own models:

1. Prepare training data in the `data/training_data/` directory
2. Use the training scripts in the `legacy/train_model/` directory
3. Save the trained models in this directory with the expected filenames

## Model Performance

When models are available, the system provides:
- Risk assessment with confidence scores
- Goal success probability predictions
- Feature importance analysis
- Model performance metrics