# ML Data Directory

This directory contains training datasets for the Prospect Analysis workflow ML models.

## Expected Datasets

### risk_profile_training_dataset.csv
Training dataset for risk profile prediction model.

Required columns:
- age
- annual_income
- current_savings
- investment_horizon_years
- investment_experience_level
- risk_profile (target variable: Low, Medium, High)

### goal_success_training_dataset.csv
Training dataset for goal success prediction model.

Required columns:
- current_savings
- target_goal_amount
- investment_horizon_years
- annual_income
- number_of_dependents
- goal_success (target variable: Success, Moderate, Challenging)

## Usage

Place your training datasets here and run the training script:

```bash
python -m ml.training.train_models
```

The trained models will be saved to `ml/models/` directory.
