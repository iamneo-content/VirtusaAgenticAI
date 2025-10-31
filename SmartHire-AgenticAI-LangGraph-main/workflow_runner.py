import time
import streamlit as st
from graph import app
from state import CandidateState

def run_workflow_with_progress(initial_state: CandidateState):
    """Run workflow with progress tracking and timeout handling"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Resume Parsing
        status_text.text("🔄 Step 1/5: Parsing resume with AI...")
        progress_bar.progress(20)
        time.sleep(0.5)  # Small delay for UI update
        
        # Step 2: Experience Prediction
        status_text.text("🔄 Step 2/5: Predicting experience level...")
        progress_bar.progress(40)
        time.sleep(0.5)
        
        # Step 3: Resume Scoring
        status_text.text("🔄 Step 3/5: Scoring resume quality...")
        progress_bar.progress(60)
        time.sleep(0.5)
        
        # Step 4: Job Fit Analysis
        status_text.text("🔄 Step 4/5: Analyzing job fit...")
        progress_bar.progress(80)
        time.sleep(0.5)
        
        # Step 5: Question Generation
        status_text.text("🔄 Step 5/5: Generating interview questions...")
        progress_bar.progress(90)
        
        # Run the actual workflow
        result = app.invoke(initial_state)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Analysis completed successfully!")
        time.sleep(1)
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return result, None
        
    except Exception as e:
        # Clean up progress indicators on error
        progress_bar.empty()
        status_text.empty()
        
        return None, str(e)