import os
import unittest
import pandas as pd
from state import CandidateState
from nodes.parse_resume import parse_resume
from ml.training.predict_experience import train_experience_model, predict_experience
from ml.training.score_resume import train_resume_score_model, predict_resume_score
from nodes.analyze_job_fit import analyze_job_fit
from nodes.generate_questions import generate_questions
from nodes.evaluate_answers import evaluate_answers
from utils.pdf_extractor import extract_text_from_pdf
from graph import create_workflow
from dotenv import load_dotenv

load_dotenv()

class TestSmartHire(unittest.TestCase):
    
    def setUp(self):
        """Setup test data"""
        self.sample_resume_text = """
        John Doe
        john.doe@email.com
        5 years of experience in software development
        Skills: Python, JavaScript, SQL, React, Node.js
        Projects: E-commerce platform, Data analysis tool
        Education: B.S. Computer Science, ABC University
        """
        
        self.sample_job_description = """
        We are looking for a Software Engineer with experience in Python and JavaScript.
        Requirements: 3-5 years of experience, knowledge of web frameworks, and database management.
        """
        
        self.initial_state = CandidateState(
            resume_text=self.sample_resume_text,
            job_description=self.sample_job_description,
            job_title="Software Engineer",
            candidate_name="",
            resume_features=None,
            experience_level=None,
            resume_score=None,
            job_fit=None,
            questions=[],
            answers=[],
            final_score=None,
            feedback=None,
            current_step="start",
            errors=[]
        )

    def test_api_keys_loaded(self):
        """Test that all Gemini API keys are properly loaded"""
        api_keys = [
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"), 
            os.getenv("GEMINI_API_KEY_3"),
            os.getenv("GEMINI_API_KEY_4")
        ]
        for key in api_keys:
            self.assertIsNotNone(key, "API key should be loaded from .env")
            self.assertTrue(key.startswith("AIza"), "API key should start with 'AIza'")

    def test_resume_parsing(self):
        """Test resume parsing functionality"""
        state_with_resume = self.initial_state.copy()
        state_with_resume['resume_text'] = self.sample_resume_text
        updated_state = parse_resume(state_with_resume)
        
        self.assertIsNotNone(updated_state['resume_features'], "Resume features should not be none")
        self.assertIn('skills', updated_state['resume_features'], "Skills should be extracted")
        self.assertIn('full_name', updated_state['resume_features'], "Name should be extracted")
        self.assertIn('education', updated_state['resume_features'], "Education should be extracted")
        self.assertEqual(updated_state['current_step'], 'parsed', "Current step should be 'parsed'")

    def test_experience_model_training(self):
        """Test experience prediction model training"""
        success = train_experience_model()
        self.assertTrue(success, "Experience model training should succeed")

        # Verify model files are created
        model_file = "ml/models/experience_level_model.pkl"
        encoder_file = "ml/models/experience_level_encoder.pkl"
        features_file = "ml/models/experience_features.pkl"

        self.assertTrue(os.path.exists(model_file), "Experience model file should exist")
        self.assertTrue(os.path.exists(encoder_file), "Experience encoder file should exist")
        self.assertTrue(os.path.exists(features_file), "Experience features file should exist")

    def test_resume_model_training(self):
        """Test resume scoring model training"""
        success = train_resume_score_model()
        self.assertTrue(success, "Resume scoring model training should succeed")

        # Verify model files are created
        model_file = "ml/models/resume_score_model.pkl"
        scaler_file = "ml/models/resume_score_scaler.pkl"
        features_file = "ml/models/resume_score_features.pkl"

        self.assertTrue(os.path.exists(model_file), "Resume model file should exist")
        self.assertTrue(os.path.exists(scaler_file), "Resume scaler file should exist")
        self.assertTrue(os.path.exists(features_file), "Resume features file should exist")

    def test_experience_prediction(self):
        """Test experience prediction after training"""
        # First ensure model is trained
        train_experience_model()
        
        # Set up state with resume features
        experience_state = self.initial_state.copy()
        experience_state['resume_features'] = {
            'total_experience_years': 5.0,
            'skills': ['Python', 'JavaScript'],
            'projects': ['Project 1', 'Project 2'],
            'leadership_experience': 1
        }
        
        updated_state = predict_experience(experience_state)
        self.assertIsNotNone(updated_state['experience_level'], "Experience level should be predicted")
        self.assertIn(updated_state['experience_level'], ['Junior', 'Mid-level', 'Senior'], 
                     "Experience level should be one of the expected values")
        self.assertEqual(updated_state['current_step'], 'experience_predicted', 
                        "Current step should be 'experience_predicted'")

    def test_resume_scoring(self):
        """Test resume scoring after training"""
        # First ensure model is trained
        train_resume_score_model()
        
        # Set up state with resume features
        scoring_state = self.initial_state.copy()
        scoring_state['resume_features'] = {
            'total_experience_years': 5.0,
            'skills': ['Python', 'JavaScript', 'React', 'Node.js'],
            'projects': ['Project 1', 'Project 2', 'Project 3'],
            'certifications': ['AWS', 'Python'],
            'leadership_experience': 1,
            'has_research_work': 0
        }
        
        updated_state = predict_resume_score(scoring_state)
        self.assertIsNotNone(updated_state['resume_score'], "Resume score should be predicted")
        self.assertGreaterEqual(updated_state['resume_score'], 0, "Resume score should be >= 0")
        self.assertLessEqual(updated_state['resume_score'], 10, "Resume score should be <= 10")

    def test_job_fit_analysis(self):
        """Test job fit analysis"""
        state = self.initial_state.copy()
        state['resume_features'] = {
            'skills': ['Python', 'JavaScript', 'SQL'],
            'total_experience_years': 3.0
        }
        
        updated_state = analyze_job_fit(state)
        self.assertIsNotNone(updated_state['job_fit'], "Job fit should be analyzed")
        self.assertIn('job_fit', updated_state['job_fit'], "Job fit result should be present")
        self.assertIn('fit_score', updated_state['job_fit'], "Fit score should be present")
        self.assertIn('reason', updated_state['job_fit'], "Fit reason should be present")

    def test_question_generation(self):
        """Test question generation"""
        state = self.initial_state.copy()
        state['resume_features'] = {
            'skills': ['Python', 'JavaScript'],
            'total_experience_years': 3.0
        }
        state['experience_level'] = 'Mid-level'
        state['job_title'] = 'Software Engineer'
        
        updated_state = generate_questions(state)
        self.assertIsNotNone(updated_state['questions'], "Questions should be generated")
        self.assertGreater(len(updated_state['questions']), 0, "At least one question should be generated")
        for question in updated_state['questions']:
            self.assertIn('question', question, "Each question should have a question text")
            self.assertIn('type', question, "Each question should have a type")
            self.assertIn('reference_answer', question, "Each question should have a reference answer")

    def test_answer_evaluation(self):
        """Test answer evaluation"""
        state = self.initial_state.copy()
        state['questions'] = [
            {'id': 'Q1', 'type': 'concept', 'question': 'Explain OOP', 'reference_answer': 'Object-oriented programming...'},
            {'id': 'Q2', 'type': 'code', 'question': 'Write a function', 'reference_answer': 'def example(): ...'}
        ]
        state['answers'] = [
            {'answer': 'Object-oriented programming involves classes and objects...'},
            {'answer': 'def my_function(): return "Hello"'}
        ]
        
        updated_state = evaluate_answers(state)
        self.assertIsNotNone(updated_state['final_score'], "Final score should be calculated")
        self.assertGreaterEqual(updated_state['final_score'], 0, "Final score should be >= 0")
        self.assertLessEqual(updated_state['final_score'], 10, "Final score should be <= 10")
        self.assertEqual(len(updated_state['answers']), 2, "All answers should be evaluated")

    def test_workflow_creation(self):
        """Test LangGraph workflow creation"""
        workflow = create_workflow()
        self.assertIsNotNone(workflow, "Workflow should be created successfully")

if __name__ == '__main__':
    # Run the tests and print results
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSmartHire)
    
    # Count total tests
    total_tests = suite.countTestCases()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {total_tests - len(result.failures) - len(result.errors)}")
    print(f"Tests Failed: {len(result.failures)}")
    print(f"Tests with Errors: {len(result.errors)}")
    print(f"Tests Skipped: {len(result.skipped)}")
    print("="*50)