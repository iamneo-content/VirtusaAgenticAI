# ML Module - Machine Learning Components

This directory contains all machine learning-related components for the SmartHire system, including model training utilities, trained models, and ML prediction agents.

## Directory Structure

```
ml/
├── agents/              # ML prediction agents
│   ├── __init__.py
│   ├── experience_predictor.py    # Experience level classification (RandomForest)
│   └── resume_scorer.py           # Resume scoring regression (RandomForest)
│
├── training/            # Model training entry point
│   ├── __init__.py
│   └── train_models.py            # Main training orchestration script
│
├── models/              # Trained ML models (generated after training)
│   ├── experience_level_model.pkl
│   ├── experience_level_encoder.pkl
│   ├── experience_features.pkl
│   ├── resume_score_model.pkl
│   ├── resume_score_scaler.pkl
│   └── resume_score_features.pkl
│
├── __init__.py
└── README.md

Note: Training datasets are located in the main ../data/ folder, not in ml/
```

## Components

### ML Agents (ml/agents/)

#### experience_predictor.py
- **Model Type:** RandomForestClassifier
- **Task:** Predict candidate experience level (Junior/Mid-level/Senior)
- **Input Features:** total_experience_years, skills_count, project_count, leadership_experience
- **Output:** Experience level category
- **Key Functions:**
  - `train_experience_model()` - Train the model on dataset
  - `predict_experience(state)` - Make predictions for a candidate

#### resume_scorer.py
- **Model Type:** RandomForestRegressor
- **Task:** Score resume quality on a 0-10 scale
- **Input Features:** total_experience_years, skills_count, project_count, certification_count, leadership_experience, has_research_work
- **Output:** Resume score (0-10)
- **Key Functions:**
  - `train_resume_score_model()` - Train the model on dataset
  - `predict_resume_score(state)` - Generate resume scores

### Training Module (ml/training/)

#### train_models.py
Main orchestration script that imports training functions from ml/agents/ and trains all ML models. Run this before first use:

```bash
python ml/training/train_models.py
```

**Note:** This script imports training functions from:
- `ml.agents.experience_predictor.train_experience_model()`
- `ml.agents.resume_scorer.train_resume_score_model()`

There are no separate legacy trainers - all training logic is in the agent files themselves.

### Training Data (data/ - Root Directory)

Training datasets are stored in the main `data/` folder to keep them separate from ML models.

#### data/experience_level_training_dataset.csv
Training data for experience level classification.
- Columns: total_experience_years, skills_count, project_count, leadership_experience, experience_level
- Target Classes: Junior, Mid-level, Senior

#### data/resume_score_training_dataset.csv
Training data for resume scoring regression.
- Columns: total_experience_years, skills_count, project_count, certification_count, leadership_experience, has_research_work, resume_score
- Target Range: 0-10 (continuous)

#### data/resume_fit_training_dataset.csv
Training data for job-candidate fit classification.
- Columns: candidate_id, job_id, total_experience_years, education_level, skills_list, certification_count, project_count, job_fit_label
- Note: Currently not used in main pipeline but available for future enhancements

## Usage

### Training Models

Before using the application for the first time, train the ML models:

```bash
# From project root directory
python ml/training/train_models.py
```

This will generate trained model files in `ml/models/`.

### Using ML Agents in Workflow

The ML agents are integrated into the main LangGraph workflow:

```python
from ml.agents import predict_experience, predict_resume_score

# In workflow.py
from langgraph.graph import StateGraph
from ml.agents.experience_predictor import predict_experience
from ml.agents.resume_scorer import predict_resume_score

workflow = StateGraph(CandidateState)
workflow.add_node("predict_experience", predict_experience)
workflow.add_node("score_resume", predict_resume_score)
```

### Direct Prediction

You can also use the agents directly:

```python
from ml.agents import predict_experience, predict_resume_score
from state import CandidateState

# Create candidate state with resume features
state = CandidateState(
    resume_features={
        'total_experience_years': 5.0,
        'skills': ['Python', 'JavaScript'],
        'projects': ['Project 1', 'Project 2'],
        'leadership_experience': 1
    },
    ...
)

# Make predictions
state = predict_experience(state)
state = predict_resume_score(state)

print(f"Experience Level: {state['experience_level']}")
print(f"Resume Score: {state['resume_score']}")
```

## Model Details

### Experience Level Prediction
- **Algorithm:** Random Forest Classifier (100 estimators)
- **Training Data:** experience_level_training_dataset.csv
- **Metrics:** Evaluated on test set (20% split)
- **Fallback:** Rule-based heuristics based on years of experience

### Resume Scoring
- **Algorithm:** Random Forest Regressor (100 estimators)
- **Preprocessing:** StandardScaler normalization
- **Training Data:** resume_score_training_dataset.csv
- **Output Range:** 0-10 (clipped)
- **Fallback:** Component-based scoring (experience, skills, projects, etc.)

## Fallback Mechanisms

Both ML agents have built-in fallback mechanisms:

1. **Model Not Found:** If trained models don't exist, the system attempts to train them automatically
2. **Training Fails:** If model training fails, rule-based fallbacks are used
3. **Prediction Errors:** Any prediction errors trigger fallback logic with error logging

## Adding Custom Models

To add new ML models to this module:

1. Create a new predictor agent in `ml/agents/your_model.py` with both `train_your_model()` and `predict_your_model(state)` functions
2. Add training datasets to `data/` (root directory)
3. Update `ml/training/train_models.py` to import and call your training function
4. Update `ml/__init__.py` to export your new agent functions
5. Update `workflow.py` to integrate the new agent node

**Note:** There's no separate training folder for individual models - training functions live in the agent files themselves (ml/agents/)

## Dependencies

The ML module requires:
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- joblib >= 1.3.0

## Performance Monitoring

Model performance metrics are logged during training:
- Experience Prediction: Classification report
- Resume Scoring: R² Score and Mean Squared Error (MSE)

## Updating Training Data

To improve model performance, update the training datasets in `ml/data/`:

1. Add new training examples to the CSV files
2. Run `python ml/training/train_models.py` to retrain
3. Verify models are saved in `ml/models/`

## Troubleshooting

### Models Not Training
- Check that training data files exist in `data/` folder (root directory)
- Verify CSV format and required columns
- Check console output for specific errors

### Poor Predictions
- Review training data quality and completeness
- Consider updating training datasets with more examples
- Check that input feature extraction is correct

### Path Errors
- Ensure scripts are run from project root directory
- Verify all paths use forward slashes (/) or raw strings
- Check that ml/ directory and subdirectories exist
- Verify that `data/` folder contains the training CSV files
