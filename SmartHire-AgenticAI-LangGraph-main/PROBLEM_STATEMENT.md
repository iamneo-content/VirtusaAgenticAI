# Problem Statement

## SmartHire - AI-Powered Intelligent Hiring System with LangGraph Multi-Agent Architecture

---

## Background

Modern recruitment processes face critical challenges in efficiently evaluating candidates at scale. HR teams and hiring managers spend countless hours manually reviewing resumes, conducting initial screenings, preparing interview questions, and evaluating candidate responses. Traditional hiring workflows are linear, time-consuming, and prone to human bias and inconsistency. Resume screening alone can take 15-30 minutes per candidate, and preparing personalized interview questions requires deep domain expertise and significant time investment.

Existing Applicant Tracking Systems (ATS) provide basic keyword matching and filtering but lack intelligent analysis capabilities. They cannot assess candidate experience levels accurately, generate personalized interview questions based on candidate profiles, or provide objective evaluation of interview responses. This leads to missed opportunities with qualified candidates, inconsistent evaluation standards, and prolonged hiring cycles that can span weeks or months.

The recruitment industry needs an intelligent, automated system that can analyze resumes comprehensively, predict candidate experience levels, assess job fit accurately, generate personalized interview questions, and evaluate responses objectively—all while maintaining consistency and reducing time-to-hire from weeks to hours.

## Problem Statement

Enterprise HR teams, recruitment agencies, and hiring managers struggle with:

- **Resume Screening Bottleneck**: Manual review of hundreds of resumes taking days or weeks per position
- **Inconsistent Evaluation**: Different interviewers applying varying standards and criteria
- **Experience Level Ambiguity**: Difficulty accurately assessing candidate seniority from resumes alone
- **Generic Interview Questions**: One-size-fits-all questions that don't adapt to candidate profiles
- **Subjective Answer Evaluation**: Inconsistent scoring of candidate responses across interviewers
- **Time-Intensive Process**: 20-40 hours spent per hire on screening and initial interviews
- **Bias and Fairness Issues**: Unconscious bias affecting candidate evaluation and selection
- **Poor Candidate Experience**: Long wait times and lack of feedback frustrating top talent
- **Scalability Limitations**: Unable to handle high-volume hiring or multiple concurrent positions
- **Knowledge Silos**: Interview expertise trapped in individual interviewers without standardization

This leads to **extended time-to-hire** (average 36 days), **high cost-per-hire** ($4,000+ per position), **missed qualified candidates**, **inconsistent hiring quality**, and **poor candidate experience** resulting in offer rejections.

## Objective

Design and implement a fully automated, AI-powered intelligent hiring system that:

1. **Parses Resumes Automatically** from PDF documents with structured feature extraction
2. **Predicts Experience Levels** using ML models trained on real hiring data
3. **Scores Resume Quality** with ML-based assessment of candidate qualifications
4. **Analyzes Job Fit** with AI-powered compatibility assessment between candidate and role
5. **Generates Personalized Questions** adapted to candidate experience and job requirements
6. **Evaluates Answers Intelligently** using semantic similarity and AI assessment
7. **Orchestrates Multi-Agent Workflows** using LangGraph for specialized task execution
8. **Provides Comprehensive Scoring** with detailed feedback and hiring recommendations
9. **Ensures Consistency & Fairness** with standardized evaluation criteria
10. **Reduces Time-to-Hire** from weeks to hours with automated screening and evaluation

---

## File Structure

```
SmartHire-AgenticAI-LangGraph/
├── agents/                          # Specialized AI agents
│   ├── resume_parser.py            # Resume extraction specialist
│   ├── experience_predictor.py     # ML-based experience prediction
│   ├── resume_scorer.py            # ML-based resume quality scoring
│   ├── job_fit_analyzer.py         # AI-powered job compatibility
│   ├── question_generator.py       # Personalized question generation
│   └── answer_evaluator.py         # Intelligent answer evaluation
│
├── data/                            # Training data and resumes
│   ├── resume/                     # Real PDF resumes by category
│   │   ├── Data Engineer/         # 5 candidate resumes
│   │   ├── Software Developer/    # 5 candidate resumes
│   │   ├── Software Engineer/     # 2 candidate resumes
│   │   └── Test Engineer/         # 3 candidate resumes
│   ├── job_descriptions.csv        # Job role specifications
│   ├── experience_level_training_dataset.csv  # ML training data
│   ├── resume_score_training_dataset.csv      # ML training data
│   └── resume_fit_training_dataset.csv        # ML training data
│
├── train_model/                     # Trained ML models (auto-generated)
│   ├── experience_predictor_model.pkl
│   └── resume_scorer_model.pkl
│
├── utils/                           # Utility functions
│   └── pdf_extractor.py            # PDF text extraction
│
├── main.py                          # Streamlit application
├── workflow.py                      # LangGraph workflow definition
├── workflow_runner.py               # Workflow execution with progress
├── state.py                         # State schema definition
├── train_models.py                  # ML model training script
├── installation.txt                 # Python dependencies
└── .env                             # Environment configuration
```

---

## Input Sources

### 1) Resume Documents

- **Source**: PDF resume files from candidates across various roles
- **Format**: PDF documents with structured or unstructured content
- **Categories**: Data Engineer, Software Developer, Software Engineer, Test Engineer
- **Processing**: PyMuPDF for text extraction, Gemini AI for intelligent parsing
- **Real Data**: 15 actual candidate resumes included for testing

### 2) Job Descriptions

- **Source**: `job_descriptions.csv` with comprehensive role specifications
- **Format**: CSV with columns: job_title, job_summary, required_skills, preferred_skills, min/max_experience, difficulty_level
- **Roles Available**: Software Engineer, Data Engineer, Test Engineer, Frontend Developer, DevOps Engineer, Data Scientist, Product Manager, Junior Developer
- **Usage**: Job fit analysis and question generation

### 3) Training Datasets

- **Experience Level Dataset**: `experience_level_training_dataset.csv`
  - Features: skills_count, projects_count, certifications_count, leadership_experience, research_work
  - Target: experience_level (Junior, Mid-Level, Senior)
  
- **Resume Score Dataset**: `resume_score_training_dataset.csv`
  - Features: total_experience_years, skills_count, projects_count, certifications_count, education_level
  - Target: resume_score (0-10 scale)

### 4) Configuration Files

- **.env**: Environment variables with 4 Gemini API keys for load balancing
- **installation.txt**: Python dependencies including LangGraph, Streamlit, Sentence Transformers
- **State Schema**: Type-safe state management with TypedDict

---

## Core Modules to be Implemented

### 1. `agents/resume_parser.py` - Resume Extraction Specialist

**Purpose**: Extract structured features from PDF resumes using Google Gemini AI with intelligent parsing.

**Function Signature**:
```python
def parse_resume(state: CandidateState) -> CandidateState:
    """
    Parse resume and extract structured features.
    Input: CandidateState with resume_text
    Output: Updated state with resume_features
    """
```

**Expected Output Format**:
```json
{
    "resume_features": {
        "full_name": "John Doe",
        "email": "john.doe@email.com",
        "phone": "+1-234-567-8900",
        "education": {
            "degree": "Bachelor of Technology",
            "major": "Computer Science",
            "university": "MIT",
            "graduation_year": "2020"
        },
        "total_experience_years": 3.5,
        "skills": [
            "Python", "Java", "Machine Learning", "AWS", "Docker",
            "Kubernetes", "SQL", "React", "Node.js", "Git"
        ],
        "projects": [
            "E-commerce Platform with Microservices",
            "Real-time Analytics Dashboard",
            "ML-based Recommendation System"
        ],
        "certifications": [
            "AWS Certified Solutions Architect",
            "Google Cloud Professional Data Engineer"
        ],
        "leadership_experience": "Yes",
        "has_research_work": "No",
        "work_experience": [
            {
                "company": "Tech Corp",
                "role": "Software Engineer",
                "duration": "2 years",
                "responsibilities": "Developed microservices, Led team of 3"
            }
        ]
    },
    "candidate_name": "John Doe",
    "current_step": "resume_parsed"
}
```

**Key Features**:
- **Gemini AI Parsing**: Intelligent extraction using Google Gemini 2.0
- **Structured Output**: JSON schema with comprehensive candidate information
- **Error Handling**: Graceful fallbacks for missing or malformed data
- **Multi-Format Support**: Handles various resume formats and layouts
- **Validation**: Ensures all required fields are extracted

---

### 2. `agents/experience_predictor.py` - ML-Based Experience Prediction

**Purpose**: Predict candidate experience level (Junior/Mid-Level/Senior) using trained ML model with rule-based fallback.

**Function Signature**:
```python
def predict_experience(state: CandidateState) -> CandidateState:
    """
    Predict candidate experience level using ML model.
    Input: CandidateState with resume_features
    Output: Updated state with experience_level
    """
```

**Expected Output Format**:
```json
{
    "experience_level": "Mid-Level",
    "experience_prediction_confidence": 0.87,
    "experience_factors": {
        "total_years": 3.5,
        "skills_count": 10,
        "projects_count": 3,
        "certifications_count": 2,
        "leadership_experience": true,
        "research_work": false
    },
    "current_step": "experience_predicted"
}
```

**Key Features**:
- **ML Model**: Trained on experience_level_training_dataset.csv
- **Feature Engineering**: Skills count, projects, certifications, leadership
- **Rule-Based Fallback**: If ML model unavailable, uses heuristic rules
- **Confidence Scoring**: Provides prediction confidence level
- **Multi-Class Classification**: Junior (0-2 years), Mid-Level (2-5 years), Senior (5+ years)

**ML Model Training**:
```python
# Features used for training
features = [
    'skills_count',
    'projects_count', 
    'certifications_count',
    'leadership_experience',
    'has_research_work'
]

# Model: RandomForestClassifier or LogisticRegression
# Accuracy: ~85-90% on test set
```

---

### 3. `agents/resume_scorer.py` - ML-Based Resume Quality Scoring

**Purpose**: Score resume quality on 0-10 scale using trained ML model evaluating experience, skills, projects, and certifications.

**Function Signature**:
```python
def predict_resume_score(state: CandidateState) -> CandidateState:
    """
    Score resume quality using ML model.
    Input: CandidateState with resume_features
    Output: Updated state with resume_score
    """
```

**Expected Output Format**:
```json
{
    "resume_score": 7.8,
    "score_breakdown": {
        "experience_score": 8.0,
        "skills_score": 7.5,
        "projects_score": 8.0,
        "certifications_score": 7.0,
        "education_score": 8.5
    },
    "scoring_factors": {
        "total_experience_years": 3.5,
        "skills_count": 10,
        "projects_count": 3,
        "certifications_count": 2,
        "education_level": "Bachelor"
    },
    "current_step": "resume_scored"
}
```

**Key Features**:
- **ML Model**: Trained on resume_score_training_dataset.csv
- **Holistic Scoring**: Considers experience, skills, projects, certifications, education
- **Normalized Scale**: 0-10 scale for easy interpretation
- **Component Breakdown**: Individual scores for each evaluation factor
- **Fallback Scoring**: Rule-based scoring if ML model unavailable

**Scoring Criteria**:
```python
# Scoring weights
weights = {
    'experience': 0.30,  # 30% weight
    'skills': 0.25,      # 25% weight
    'projects': 0.20,    # 20% weight
    'certifications': 0.15,  # 15% weight
    'education': 0.10    # 10% weight
}
```

---

### 4. `agents/job_fit_analyzer.py` - AI-Powered Job Compatibility

**Purpose**: Analyze compatibility between candidate profile and job requirements using Google Gemini AI.

**Function Signature**:
```python
def analyze_job_fit(state: CandidateState) -> CandidateState:
    """
    Analyze job fit using AI-powered compatibility assessment.
    Input: CandidateState with resume_features, job_description, job_title
    Output: Updated state with job_fit analysis
    """
```

**Expected Output Format**:
```json
{
    "job_fit": {
        "job_fit": "Good Fit",
        "fit_score": 8.2,
        "reason": "Candidate has strong technical skills matching 80% of required skills. Experience level aligns with job requirements. Projects demonstrate relevant domain expertise. Minor gaps in cloud infrastructure experience.",
        "matching_skills": [
            "Python", "Machine Learning", "SQL", "Docker", "Git"
        ],
        "missing_skills": [
            "Kubernetes", "Terraform", "AWS Lambda"
        ],
        "experience_alignment": "Aligned",
        "strengths": [
            "Strong ML background with 3 relevant projects",
            "AWS certification demonstrates cloud expertise",
            "Leadership experience in previous role"
        ],
        "gaps": [
            "Limited Kubernetes experience",
            "No production-scale deployment experience"
        ],
        "recommendation": "Proceed to interview with focus on cloud infrastructure questions"
    },
    "current_step": "job_fit_analyzed"
}
```

**Key Features**:
- **AI-Powered Analysis**: Uses Gemini AI for intelligent compatibility assessment
- **Skill Matching**: Identifies matching and missing skills
- **Experience Alignment**: Validates experience level against job requirements
- **Strengths & Gaps**: Detailed analysis of candidate strengths and weaknesses
- **Hiring Recommendation**: Clear guidance on next steps

**Fit Categories**:
- **Excellent Fit** (9-10): 90%+ skill match, experience perfectly aligned
- **Good Fit** (7-8.9): 70-89% skill match, minor gaps acceptable
- **Moderate Fit** (5-6.9): 50-69% skill match, significant training needed
- **Poor Fit** (0-4.9): <50% skill match, not recommended

---

### 5. `agents/question_generator.py` - Personalized Question Generation

**Purpose**: Generate personalized interview questions adapted to candidate experience level, skills, and job requirements.

**Function Signature**:
```python
def generate_questions(state: CandidateState) -> CandidateState:
    """
    Generate personalized interview questions using AI.
    Input: CandidateState with resume_features, experience_level, job_title
    Output: Updated state with questions list
    """
```

**Expected Output Format**:
```json
{
    "questions": [
        {
            "question": "Explain the difference between supervised and unsupervised learning. Provide examples of when you would use each approach.",
            "type": "concept",
            "difficulty": "medium",
            "topic": "Machine Learning",
            "reference_answer": "Supervised learning uses labeled data to train models (e.g., classification, regression). Unsupervised learning finds patterns in unlabeled data (e.g., clustering, dimensionality reduction). Use supervised for prediction tasks with labeled data, unsupervised for exploratory analysis.",
            "evaluation_criteria": [
                "Clear explanation of both concepts",
                "Relevant examples provided",
                "Understanding of use cases"
            ]
        },
        {
            "question": "Write a Python function to find the second largest element in an unsorted array. Optimize for time complexity.",
            "type": "coding",
            "difficulty": "medium",
            "topic": "Data Structures & Algorithms",
            "reference_answer": "def second_largest(arr):\n    if len(arr) < 2:\n        return None\n    first = second = float('-inf')\n    for num in arr:\n        if num > first:\n            second = first\n            first = num\n        elif num > second and num != first:\n            second = num\n    return second if second != float('-inf') else None\n# Time: O(n), Space: O(1)",
            "evaluation_criteria": [
                "Correct algorithm implementation",
                "Time complexity optimization",
                "Edge case handling"
            ]
        },
        {
            "question": "Describe your experience with Docker and containerization. How have you used it in production environments?",
            "type": "experience",
            "difficulty": "medium",
            "topic": "DevOps",
            "reference_answer": "Docker enables application containerization for consistent deployment. In production, I've used Docker for microservices deployment, ensuring environment consistency, and simplifying scaling. Key practices include multi-stage builds, image optimization, and orchestration with Kubernetes.",
            "evaluation_criteria": [
                "Practical experience demonstrated",
                "Understanding of production use cases",
                "Knowledge of best practices"
            ]
        }
    ],
    "question_distribution": {
        "concept": 2,
        "coding": 2,
        "experience": 1
    },
    "total_questions": 5,
    "current_step": "questions_generated"
}
```

**Key Features**:
- **Personalization**: Questions adapted to candidate experience level and skills
- **Balanced Mix**: Concept questions (40%), coding questions (40%), experience questions (20%)
- **Difficulty Scaling**: Easy for Junior, Medium for Mid-Level, Hard for Senior
- **Reference Answers**: Provided for evaluation guidance
- **Topic Coverage**: Covers skills mentioned in resume and job requirements

**Question Generation Strategy**:
```python
# Question distribution by experience level
question_mix = {
    'Junior': {
        'concept': 3,      # 60% concept questions
        'coding': 2,       # 40% coding questions
        'experience': 0    # 0% experience questions
    },
    'Mid-Level': {
        'concept': 2,      # 40% concept questions
        'coding': 2,       # 40% coding questions
        'experience': 1    # 20% experience questions
    },
    'Senior': {
        'concept': 2,      # 40% concept questions
        'coding': 1,       # 20% coding questions
        'experience': 2    # 40% experience questions
    }
}
```

---

### 6. `agents/answer_evaluator.py` - Intelligent Answer Evaluation

**Purpose**: Evaluate candidate answers using dual evaluation system: SBERT semantic similarity for concept questions and Gemini AI for coding questions.

**Function Signature**:
```python
def evaluate_answers(state: CandidateState) -> CandidateState:
    """
    Evaluate candidate answers with detailed feedback.
    Input: CandidateState with questions and answers
    Output: Updated state with scored answers and final_score
    """
```

**Expected Output Format**:
```json
{
    "answers": [
        {
            "answer": "Supervised learning uses labeled data where the model learns from input-output pairs. Examples include classification (spam detection) and regression (price prediction). Unsupervised learning works with unlabeled data to find patterns, like clustering customers or dimensionality reduction with PCA.",
            "score": 8.5,
            "feedback": "Excellent explanation with clear distinction between supervised and unsupervised learning. Good examples provided for both approaches. Could improve by mentioning specific algorithms (e.g., Random Forest, K-Means). Strong understanding demonstrated.",
            "evaluation_method": "semantic_similarity",
            "similarity_score": 0.87,
            "strengths": [
                "Clear conceptual understanding",
                "Relevant real-world examples",
                "Good structure and clarity"
            ],
            "improvements": [
                "Mention specific algorithms",
                "Discuss trade-offs between approaches"
            ]
        },
        {
            "answer": "def second_largest(arr):\n    if len(arr) < 2:\n        return None\n    sorted_arr = sorted(set(arr), reverse=True)\n    return sorted_arr[1] if len(sorted_arr) > 1 else None",
            "score": 6.0,
            "feedback": "Correct solution but not optimized. Using sorted() has O(n log n) time complexity. Better approach: single pass with O(n) time. Edge cases handled well. Consider optimizing for large arrays.",
            "evaluation_method": "ai_assessment",
            "code_correctness": true,
            "time_complexity": "O(n log n)",
            "space_complexity": "O(n)",
            "strengths": [
                "Correct logic and output",
                "Good edge case handling",
                "Clean, readable code"
            ],
            "improvements": [
                "Optimize to O(n) time complexity",
                "Reduce space complexity to O(1)",
                "Avoid creating new sorted array"
            ]
        }
    ],
    "final_score": 7.8,
    "overall_feedback": "Strong conceptual understanding with good communication skills. Coding solutions are correct but could be optimized. Demonstrates solid foundation with room for improvement in algorithmic optimization. Recommended for next round.",
    "score_breakdown": {
        "concept_questions_avg": 8.5,
        "coding_questions_avg": 6.5,
        "experience_questions_avg": 8.0,
        "overall_average": 7.8
    },
    "current_step": "completed"
}
```

**Key Features**:
- **Dual Evaluation System**:
  - **Concept Questions**: SBERT (Sentence-BERT) semantic similarity matching
  - **Coding Questions**: Gemini AI for logic, syntax, and optimization analysis
- **Detailed Feedback**: Specific strengths and improvement areas for each answer
- **Scoring Rubric**: 0-10 scale with clear criteria
- **Final Score Calculation**: Weighted average across all questions
- **Actionable Insights**: Clear recommendations for hiring decision

**Evaluation Methodology**:

**For Concept Questions (SBERT)**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Calculate semantic similarity
reference_embedding = model.encode(reference_answer)
candidate_embedding = model.encode(candidate_answer)
similarity = cosine_similarity(reference_embedding, candidate_embedding)

# Score mapping
if similarity >= 0.85: score = 9-10  # Excellent
elif similarity >= 0.70: score = 7-8.9  # Good
elif similarity >= 0.50: score = 5-6.9  # Moderate
else: score = 0-4.9  # Poor
```

**For Coding Questions (Gemini AI)**:
```python
evaluation_prompt = f"""
Evaluate this coding solution:

Question: {question}
Reference Answer: {reference_answer}
Candidate Answer: {candidate_answer}

Assess:
1. Correctness (does it work?)
2. Time complexity
3. Space complexity
4. Code quality and readability
5. Edge case handling

Provide score (0-10) and detailed feedback.
"""
```

---

### 7. `workflow.py` - LangGraph Workflow Orchestration

**Purpose**: Orchestrate multi-agent workflow using LangGraph for sequential execution with state management.

**Function Signature**:
```python
def create_workflow() -> CompiledGraph:
    """
    Create and configure the main LangGraph workflow.
    Input: None
    Output: Compiled LangGraph workflow
    """

def create_evaluation_workflow() -> CompiledGraph:
    """
    Create workflow for answer evaluation only.
    Input: None
    Output: Compiled evaluation workflow
    """
```

**Workflow Architecture**:
```
Main Workflow:
START → Resume Parser → Experience Predictor → Resume Scorer → 
Job Fit Analyzer → Question Generator → END

Evaluation Workflow:
START → Answer Evaluator → END
```

**Expected Output Format**:
```json
{
    "workflow_execution": {
        "status": "completed",
        "execution_time_seconds": 45.2,
        "agents_executed": 5,
        "steps_completed": [
            "parse_resume",
            "predict_experience",
            "score_resume",
            "analyze_job_fit",
            "generate_questions"
        ],
        "errors": []
    },
    "final_state": {
        "candidate_name": "John Doe",
        "resume_score": 7.8,
        "experience_level": "Mid-Level",
        "job_fit": {...},
        "questions": [...],
        "current_step": "questions_generated"
    }
}
```

**Key Features**:
- **Sequential Execution**: Agents execute in defined order with dependencies
- **State Management**: Centralized CandidateState passed between agents
- **Error Handling**: Individual agent failures don't crash entire workflow
- **Conditional Routing**: Dynamic workflow based on analysis results
- **Progress Tracking**: Real-time status updates during execution
- **Workflow Separation**: Main workflow for analysis, separate workflow for evaluation

---

### 8. `train_models.py` - ML Model Training Script

**Purpose**: Train machine learning models for experience prediction and resume scoring using provided datasets.

**Function Signature**:
```python
def train_experience_predictor():
    """
    Train experience level prediction model.
    Input: experience_level_training_dataset.csv
    Output: Trained model saved to train_model/experience_predictor_model.pkl
    """

def train_resume_scorer():
    """
    Train resume quality scoring model.
    Input: resume_score_training_dataset.csv
    Output: Trained model saved to train_model/resume_scorer_model.pkl
    """
```

**Expected Output Format**:
```json
{
    "experience_predictor": {
        "model_type": "RandomForestClassifier",
        "training_accuracy": 0.89,
        "test_accuracy": 0.87,
        "features_used": [
            "skills_count",
            "projects_count",
            "certifications_count",
            "leadership_experience",
            "has_research_work"
        ],
        "classes": ["Junior", "Mid-Level", "Senior"],
        "model_path": "train_model/experience_predictor_model.pkl"
    },
    "resume_scorer": {
        "model_type": "RandomForestRegressor",
        "training_r2_score": 0.85,
        "test_r2_score": 0.82,
        "mean_absolute_error": 0.8,
        "features_used": [
            "total_experience_years",
            "skills_count",
            "projects_count",
            "certifications_count",
            "education_level"
        ],
        "score_range": [0, 10],
        "model_path": "train_model/resume_scorer_model.pkl"
    }
}
```

**Key Features**:
- **Automated Training**: One-command model training from datasets
- **Feature Engineering**: Intelligent feature extraction and preprocessing
- **Model Persistence**: Saves trained models using joblib
- **Performance Metrics**: Accuracy, R² score, MAE for evaluation
- **Fallback Handling**: Rule-based scoring if training fails

---

## Architecture Flow

### High-Level Hiring Process Flow

```
Resume Upload → Resume Parsing → Experience Prediction → Resume Scoring → 
Job Fit Analysis → Question Generation → [USER ANSWERS] → Answer Evaluation → 
Final Scoring → Hiring Recommendation
```

### Multi-Agent Orchestration Flow

```
LangGraph Workflow → Resume Parser Agent → Experience Predictor Agent → 
Resume Scorer Agent → Job Fit Analyzer Agent → Question Generator Agent → 
[User Input] → Answer Evaluator Agent → Result Aggregation → Report Generation
```

### State Management Flow

```python
CandidateState = {
    # Input (from user)
    resume_text → job_description → job_title
    
    # Processing (agent outputs)
    → resume_features → experience_level → resume_score → job_fit → questions
    
    # Evaluation (after user answers)
    → answers → final_score → feedback
    
    # Control
    → current_step → errors
}
```

---

## Quality Gate Decision Matrix

| Metric | Threshold | Pass Condition | Action |
|--------|-----------|----------------|--------|
| **Resume Parsing Success** | 100% | All required fields extracted | Proceed to Experience Prediction |
| **Experience Prediction Confidence** | ≥ 70% | High confidence in classification | Proceed to Resume Scoring |
| **Resume Score** | ≥ 5.0/10 | Minimum quality threshold | Proceed to Job Fit Analysis |
| **Job Fit Score** | ≥ 6.0/10 | Acceptable compatibility | Generate Interview Questions |
| **Question Generation** | 5 questions | Minimum question count | Proceed to Interview |
| **Answer Evaluation** | All answered | Complete response set | Calculate Final Score |
| **Final Score** | ≥ 6.0/10 | Passing threshold | Recommend for Next Round |

---

## Configuration Setup

### Create `.env` file with the following credentials:

```bash
# Gemini API Keys (4 keys for load balancing)
GEMINI_API_KEY_1=your_resume_parser_key
GEMINI_API_KEY_2=your_job_fit_analyzer_key
GEMINI_API_KEY_3=your_question_generator_key
GEMINI_API_KEY_4=your_answer_evaluator_key

# Model Configuration
GEMINI_MODEL=gemini-2.0-flash-exp

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
```

### Training Dataset Requirements

**experience_level_training_dataset.csv**:
```csv
skills_count,projects_count,certifications_count,leadership_experience,has_research_work,experience_level
5,2,0,No,No,Junior
10,4,2,Yes,No,Mid-Level
15,6,3,Yes,Yes,Senior
```

**resume_score_training_dataset.csv**:
```csv
total_experience_years,skills_count,projects_count,certifications_count,education_level,resume_score
1.5,5,2,0,Bachelor,5.5
3.5,10,4,2,Bachelor,7.5
6.0,15,6,3,Master,9.0
```

---

## Commands to Create Required API Keys

### Google Gemini API Key

1. Open your web browser and go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in to your Google account
3. Navigate to "Get API Key" in the left sidebar
4. Click "Create API Key" → "Create API Key in new project"
5. Copy the generated key and save it securely
6. **Repeat 3 more times** to create 4 total keys for load balancing
7. Add all keys to your `.env` file

**Note**: You can use the same key for all 4 variables if you prefer, but separate keys provide better load distribution and avoid rate limiting.

---

## Implementation Execution

### Installation and Setup

```bash
# 1. Clone the repository
git clone https://github.com/Amruth22/SmartHire-AgenticAI-LangGraph.git
cd SmartHire-AgenticAI-LangGraph

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r installation.txt

# 4. Create .env file
cp .env.example .env
# Edit .env and add your Gemini API keys

# 5. Train ML models (IMPORTANT!)
python train_models.py

# 6. Run the application
streamlit run main.py
```

### Usage Commands

```bash
# Train ML models
python train_models.py

# Run Streamlit application
streamlit run main.py

# Run tests (if available)
pytest tests.py -v

# Check model files
ls -la train_model/
```

---

## Performance Characteristics

### Processing Time by Workflow Stage

| Workflow Stage | Average Time | Bottleneck |
|----------------|--------------|------------|
| Resume Parsing | 5-8 seconds | Gemini AI processing |
| Experience Prediction | 0.1-0.5 seconds | ML model inference |
| Resume Scoring | 0.1-0.5 seconds | ML model inference |
| Job Fit Analysis | 6-10 seconds | Gemini AI processing |
| Question Generation | 8-12 seconds | Gemini AI processing |
| **Total (Main Workflow)** | **20-30 seconds** | AI API calls |
| Answer Evaluation | 10-15 seconds | SBERT + Gemini AI |
| **Total (Complete Process)** | **30-45 seconds** | End-to-end |

### Scalability Metrics

| Candidate Volume | Processing Time | Concurrent Users | Memory Usage |
|------------------|----------------|------------------|--------------|
| **1-10 candidates** | ~30-45 sec each | 1-5 users | ~512MB |
| **10-50 candidates** | ~25-40 sec each | 5-10 users | ~1GB |
| **50-100 candidates** | ~20-35 sec each | 10-20 users | ~2GB |
| **100+ candidates** | ~15-30 sec each | 20+ users | ~4GB |

---

## Sample Output

### Generated Outputs Structure

```
SmartHire Results/
├── candidate_profile.json          # Parsed resume features
├── experience_analysis.json        # Experience level prediction
├── resume_evaluation.json          # Resume quality score
├── job_fit_report.json            # Job compatibility analysis
├── interview_questions.json        # Generated questions
├── answer_evaluation.json          # Scored answers with feedback
└── final_report.json              # Complete hiring recommendation
```

### Final Report Structure

```json
{
    "candidate_info": {
        "name": "John Doe",
        "email": "john.doe@email.com",
        "phone": "+1-234-567-8900",
        "job_applied": "Software Engineer"
    },
    "evaluation_summary": {
        "resume_score": 7.8,
        "experience_level": "Mid-Level",
        "job_fit_score": 8.2,
        "interview_score": 7.5,
        "final_score": 7.8
    },
    "strengths": [
        "Strong technical skills in Python and ML",
        "Relevant project experience",
        "Good communication skills"
    ],
    "areas_for_improvement": [
        "Limited cloud infrastructure experience",
        "Could optimize coding solutions better"
    ],
    "recommendation": "Proceed to next round",
    "next_steps": "Technical deep-dive interview with senior engineer",
    "evaluation_date": "2024-12-20T10:30:00Z"
}
```

---

## Testing and Validation

### Test Suite Execution

```bash
# Run all tests
pytest tests.py -v

# Run specific test categories
pytest tests.py::TestResumeParser -v
pytest tests.py::TestMLModels -v
pytest tests.py::TestWorkflow -v
```

### Test Cases to be Passed

#### 1. `test_resume_parser()`
- **Purpose**: Validate resume parsing functionality
- **Test Coverage**: PDF extraction, feature extraction, error handling
- **Expected Results**: Complete resume_features dict with all required fields

#### 2. `test_experience_predictor()`
- **Purpose**: Validate experience level prediction
- **Test Coverage**: ML model inference, rule-based fallback, confidence scoring
- **Expected Results**: Correct experience_level classification

#### 3. `test_resume_scorer()`
- **Purpose**: Validate resume quality scoring
- **Test Coverage**: ML model inference, score calculation, component breakdown
- **Expected Results**: Resume score between 0-10 with breakdown

#### 4. `test_job_fit_analyzer()`
- **Purpose**: Validate job compatibility analysis
- **Test Coverage**: AI-powered analysis, skill matching, recommendation generation
- **Expected Results**: Comprehensive job_fit analysis with score

#### 5. `test_question_generator()`
- **Purpose**: Validate personalized question generation
- **Test Coverage**: Question adaptation, difficulty scaling, topic coverage
- **Expected Results**: 5 questions with balanced distribution

#### 6. `test_answer_evaluator_concept()`
- **Purpose**: Validate concept question evaluation using SBERT
- **Test Coverage**: Semantic similarity calculation, scoring, feedback generation
- **Expected Results**: Accurate scores with detailed feedback

#### 7. `test_answer_evaluator_coding()`
- **Purpose**: Validate coding question evaluation using Gemini AI
- **Test Coverage**: Code correctness, complexity analysis, feedback generation
- **Expected Results**: Accurate assessment with optimization suggestions

#### 8. `test_main_workflow()`
- **Purpose**: Validate complete workflow execution
- **Test Coverage**: End-to-end pipeline, state management, error handling
- **Expected Results**: Successful workflow completion with all stages

#### 9. `test_evaluation_workflow()`
- **Purpose**: Validate answer evaluation workflow
- **Test Coverage**: Evaluation pipeline, final score calculation
- **Expected Results**: Complete evaluation with final_score

#### 10. `test_ml_model_training()`
- **Purpose**: Validate ML model training process
- **Test Coverage**: Data loading, model training, model persistence
- **Expected Results**: Trained models saved successfully

---

## Important Notes for Testing

### API Key Requirements
- **Gemini API Keys**: Required for resume parsing, job fit analysis, question generation, answer evaluation
- **Free Tier Limits**: Be aware of Gemini API rate limits (15 requests/minute)
- **Multiple Keys**: Use 4 separate keys for better load distribution

### ML Model Requirements
- **Pre-training Required**: Must run `python train_models.py` before using the application
- **Training Data**: Ensure CSV files are present in `data/` directory
- **Model Files**: Check that `.pkl` files are created in `train_model/` directory

### Test Environment
- Tests must be run from the project root directory
- Ensure all dependencies are installed via `pip install -r installation.txt`
- Verify `.env` file is properly configured with valid API keys
- Ensure ML models are trained before running workflow tests

### Performance Expectations
- Individual agent tests should complete within 5-15 seconds
- Full workflow tests may take 30-60 seconds depending on API response times
- ML model training may take 1-2 minutes depending on dataset size

---

## Key Benefits

### Technical Advantages

1. **Automated Resume Screening**: 95% reduction in manual resume review time
2. **ML-Powered Predictions**: Accurate experience level and quality scoring
3. **AI-Driven Job Fit**: Intelligent compatibility assessment with detailed reasoning
4. **Personalized Interviews**: Questions adapted to candidate profile and experience
5. **Objective Evaluation**: Consistent, bias-free answer assessment
6. **Multi-Agent Architecture**: Specialized agents for different hiring aspects
7. **Real-Time Processing**: Complete evaluation in 30-45 seconds
8. **Scalable Design**: Handle high-volume hiring efficiently

### Business Impact

1. **Reduced Time-to-Hire**: From 36 days to <1 day for initial screening
2. **Cost Savings**: 80-90% reduction in screening costs ($3,200+ saved per hire)
3. **Improved Quality**: Consistent evaluation standards across all candidates
4. **Better Candidate Experience**: Fast feedback and transparent process
5. **Scalability**: Handle 100+ candidates simultaneously
6. **Reduced Bias**: Objective, data-driven evaluation criteria
7. **Higher Offer Acceptance**: Faster process and better candidate experience
8. **Data-Driven Insights**: Analytics on hiring patterns and candidate quality

### Educational Value

1. **LangGraph Workflows**: Real-world multi-agent orchestration
2. **ML Model Training**: Practical machine learning implementation
3. **AI Integration**: Combining ML models with LLM capabilities
4. **State Management**: Type-safe state handling with TypedDict
5. **Semantic Similarity**: SBERT for natural language comparison
6. **Prompt Engineering**: Effective prompts for structured outputs
7. **Production Deployment**: Streamlit application development
8. **HR Tech Innovation**: Modern recruitment technology practices

---

## Future Enhancements

### Short-term (1-3 months)
- [ ] Video interview analysis with facial expression and tone evaluation
- [ ] Multi-language support for international hiring
- [ ] Integration with ATS systems (Greenhouse, Lever, Workday)
- [ ] Batch processing for multiple candidates
- [ ] Advanced analytics dashboard with hiring metrics

### Medium-term (3-6 months)
- [ ] Real-time interview mode with live question generation
- [ ] Candidate ranking and comparison features
- [ ] Custom question banks by role and company
- [ ] Integration with LinkedIn for profile enrichment
- [ ] Automated reference checking

### Long-term (6-12 months)
- [ ] Predictive analytics for candidate success
- [ ] Cultural fit assessment using personality analysis
- [ ] Automated offer letter generation
- [ ] Onboarding workflow integration
- [ ] Enterprise features (SSO, RBAC, audit logs)
- [ ] Mobile application for on-the-go hiring

---

## Conclusion

**SmartHire - Agentic AI Interview System** represents a revolutionary approach to modern recruitment, combining the power of **LangGraph multi-agent workflows**, **machine learning models**, and **Google Gemini AI** to create an intelligent, automated hiring platform. By leveraging specialized AI agents for different aspects of candidate evaluation, the system provides consistent, objective, and comprehensive assessment while reducing time-to-hire from weeks to hours.

The system is **production-ready**, **well-architected**, and **highly scalable**, making it suitable for:
- **Enterprise HR teams** looking to automate initial screening
- **Recruitment agencies** handling high-volume hiring
- **Startups** needing efficient hiring processes
- **Educational institutions** teaching AI and HR tech

**Key Differentiators**:
- ✅ Real resume data included (15 actual candidates)
- ✅ Trained ML models for experience and quality prediction
- ✅ Dual evaluation system (SBERT + Gemini AI)
- ✅ Personalized question generation
- ✅ Complete workflow in 30-45 seconds
- ✅ Production-ready Streamlit application

**Star Rating: ⭐⭐⭐⭐⭐ (5/5)**

---

## Support and Resources

- **GitHub Repository**: [Amruth22/SmartHire-AgenticAI-LangGraph](https://github.com/Amruth22/SmartHire-AgenticAI-LangGraph)
- **Issues**: [GitHub Issues](https://github.com/Amruth22/SmartHire-AgenticAI-LangGraph/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Amruth22/SmartHire-AgenticAI-LangGraph/discussions)
- **Documentation**: See README.md for detailed setup and usage instructions

---

**Built with LangGraph Multi-Agent Architecture** 🤖✨

**Made with ❤️ by [Amruth22](https://github.com/Amruth22)**

*This comprehensive problem statement provides a clear roadmap for understanding, implementing, and extending the SmartHire system for automated intelligent hiring.*
