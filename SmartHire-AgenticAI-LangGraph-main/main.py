import streamlit as st
import pandas as pd
import json
import os
from workflow_runner import run_workflow_with_progress
from graph import evaluation_app
from state import CandidateState
from utils.pdf_extractor import extract_text_from_pdf

# Constants
RESUME_DIR = "data/resume"
JOB_DESC_PATH = "data/job_descriptions.csv"

# Load job descriptions
@st.cache_data
def load_job_data():
    return pd.read_csv(JOB_DESC_PATH)

job_df = load_job_data()

# Streamlit config
st.set_page_config(page_title="SmartHire - Agentic AI", layout="wide")
st.title("🤖 SmartHire - Agentic AI Interview System")
st.markdown("*Powered by LangGraph Multi-Agent Workflow*")

# Initialize session state
if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = None
if "current_questions" not in st.session_state:
    st.session_state.current_questions = []

# Sidebar for workflow status
with st.sidebar:
    st.header("🔄 Workflow Status")
    if st.session_state.workflow_state:
        state = st.session_state.workflow_state
        st.write(f"**Step:** {state.get('current_step', 'Not started')}")
        st.write(f"**Candidate:** {state.get('candidate_name', 'Unknown')}")
        st.write(f"**Experience:** {state.get('experience_level', 'N/A')}")
        if state.get("resume_score"):
            st.write(f"**Resume Score:** {state.get('resume_score', 0):.1f}/10")
        if state.get("job_fit"):
            st.write(f"**Job Fit:** {state['job_fit'].get('job_fit', 'N/A')}")
        if state.get("errors"):
            st.error(f"Errors: {len(state['errors'])}")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📄 Resume & Fit", "❓ Interview QA Session", "📊 Scoreboard"])

# Tab 1: Resume Analysis & Job Fit
with tab1:
    st.markdown("### Upload & Analyze Resume")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get available designations from resume directory
        designations = []
        if os.path.exists(RESUME_DIR):
            designations = [d for d in os.listdir(RESUME_DIR) if os.path.isdir(os.path.join(RESUME_DIR, d))]
        designation = st.selectbox("🎯 Select Designation", [""] + sorted(designations))
    
    with col2:
        # Get resumes for selected designation
        resumes = []
        if designation:
            resume_path = os.path.join(RESUME_DIR, designation)
            if os.path.exists(resume_path):
                resumes = [f for f in os.listdir(resume_path) if f.endswith(".pdf")]
        selected_resume = st.selectbox("📄 Select Resume", [""] + resumes)
    
    with col3:
        # Job titles from CSV
        job_titles = sorted(job_df["job_title"].unique())
        selected_job = st.selectbox("💼 Select Job Role", [""] + job_titles)

    # Show job description if job is selected
    if selected_job:
        job_row = job_df[job_df["job_title"] == selected_job]
        if not job_row.empty:
            st.markdown("#### 📋 Job Description")
            st.info(job_row.iloc[0]['job_summary'])
            
            # Show job requirements
            with st.expander("📝 Detailed Job Requirements"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Required Skills:**")
                    required_skills = job_row.iloc[0]['required_skills'].split(',')
                    for skill in required_skills:
                        st.write(f"• {skill.strip()}")
                
                with col2:
                    st.write("**Preferred Skills:**")
                    preferred_skills = job_row.iloc[0]['preferred_skills'].split(',')
                    for skill in preferred_skills:
                        st.write(f"• {skill.strip()}")
                
                st.write(f"**Experience Required:** {job_row.iloc[0]['min_required_experience']}-{job_row.iloc[0]['max_required_experience']} years")
                st.write(f"**Difficulty Level:** {job_row.iloc[0]['difficulty_level']}")

    # Analysis button
    if not designation or not selected_resume or not selected_job:
        st.warning("⚠️ Please select all three fields to proceed.")
    else:
        if st.button("🚀 Start AI Analysis", type="primary"):
            # Extract text from actual PDF
            full_path = os.path.join(RESUME_DIR, designation, selected_resume)
            
            try:
                resume_text = extract_text_from_pdf(full_path)
                
                if "[Error extracting text:" in resume_text:
                    st.error(f"❌ Failed to extract text from PDF: {resume_text}")
                    st.stop()
                
                # Get job description
                job_row = job_df[job_df["job_title"] == selected_job]
                job_description = job_row.iloc[0]["job_summary"] if not job_row.empty else "No description available"
                
                # Initialize workflow state
                initial_state = CandidateState(
                    resume_text=resume_text,
                    job_description=job_description,
                    job_title=selected_job,
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
                
                # Run workflow with progress tracking
                result, error = run_workflow_with_progress(initial_state)
                
                if error:
                    st.error(f"❌ Analysis failed: {error}")
                    st.write("**Error Details:**", error)
                else:
                    st.session_state.workflow_state = result
                    st.session_state.current_questions = result.get("questions", [])
                    st.success("✅ Analysis completed!")
                    st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Error processing resume: {str(e)}")

    # Display results if available
    if st.session_state.workflow_state:
        state = st.session_state.workflow_state
        
        if state.get("resume_features"):
            st.markdown("### 🎯 AI Evaluation Results")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Experience Level", state.get("experience_level", "N/A"))
            with col2:
                resume_score = state.get("resume_score", 0)
                st.metric("Resume Score", f"{resume_score:.1f}/10")
            with col3:
                skills_count = len(state["resume_features"].get("skills", []))
                st.metric("Skills Count", skills_count)
            with col4:
                if state.get("job_fit"):
                    fit_result = state["job_fit"].get("job_fit", "N/A")
                    st.metric("Job Fit", fit_result)

            # Job fit reason
            if state.get("job_fit"):
                fit_reason = state["job_fit"].get("reason", "No reason provided")
                st.info(f"**💡 Fit Analysis:** {fit_reason}")

            # Detailed features in expander
            with st.expander("📋 View Detailed Resume Features"):
                features = state["resume_features"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**👤 Personal Info:**")
                    st.write(f"- Name: {features.get('full_name', 'N/A')}")
                    st.write(f"- Email: {features.get('email', 'N/A')}")
                    st.write(f"- Phone: {features.get('phone', 'N/A')}")
                    
                    st.write("**🎓 Education:**")
                    education = features.get('education', {})
                    if isinstance(education, dict):
                        st.write(f"- Degree: {education.get('degree', 'N/A')}")
                        st.write(f"- Major: {education.get('major', 'N/A')}")
                        st.write(f"- University: {education.get('university', 'N/A')}")
                
                with col2:
                    st.write("**💼 Experience:**")
                    st.write(f"- Total Years: {features.get('total_experience_years', 'N/A')}")
                    st.write(f"- Leadership: {features.get('leadership_experience', 'N/A')}")
                    st.write(f"- Research Work: {features.get('has_research_work', 'N/A')}")
                    
                    st.write("**🛠️ Skills:**")
                    skills = features.get('skills', [])
                    if skills:
                        for skill in skills[:10]:  # Show first 10 skills
                            st.write(f"- {skill}")
                        if len(skills) > 10:
                            st.write(f"... and {len(skills) - 10} more")
                    else:
                        st.write("- No skills extracted")
                
                # Show projects and certifications
                st.write("**🚀 Projects:**")
                projects = features.get('projects', [])
                if projects:
                    for project in projects[:5]:  # Show first 5 projects
                        st.write(f"- {project}")
                    if len(projects) > 5:
                        st.write(f"... and {len(projects) - 5} more")
                else:
                    st.write("- No projects extracted")
                
                st.write("**🏆 Certifications:**")
                certifications = features.get('certifications', [])
                if certifications:
                    for cert in certifications:
                        st.write(f"- {cert}")
                else:
                    st.write("- No certifications extracted")

# Tab 2: Interview QA Session
with tab2:
    st.subheader("🤖 AI-Generated Interview Questions")
    
    if not st.session_state.current_questions:
        st.info("👆 Please complete resume analysis in the first tab")
    else:
        questions = st.session_state.current_questions
        st.success(f"✅ Generated {len(questions)} personalized questions")
        
        # Show questions and collect answers
        answers = []
        
        for idx, q in enumerate(questions):
            st.markdown(f"### Question {idx+1}: {q.get('type', 'General').title()} Question")
            st.markdown(f"**{q['question']}**")
            
            # Answer input
            answer_key = f"answer_{idx}"
            student_response = st.text_area(
                "💭 Your Answer:",
                key=answer_key,
                height=120,
                placeholder="Type your detailed answer here..."
            )
            answers.append({"answer": student_response})
            
            # Show reference answer in expander
            with st.expander(f"💡 Reference Answer (for guidance)"):
                st.write(q.get('reference_answer', 'No reference answer available'))
            
            st.markdown("---")
        
        # Submit button
        if st.button("📝 Submit All Answers", type="primary"):
            if all(ans["answer"].strip() for ans in answers):
                # Update state and run evaluation
                state = st.session_state.workflow_state
                state["answers"] = answers
                state["current_step"] = "answers_submitted"
                
                with st.spinner("🤖 AI is evaluating your answers..."):
                    try:
                        final_state = evaluation_app.invoke(state)
                        st.session_state.workflow_state = final_state
                        st.success("✅ Answers evaluated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Evaluation failed: {str(e)}")
            else:
                st.warning("⚠️ Please answer all questions before submitting")

# Tab 3: Scoreboard
with tab3:
    st.subheader("📊 Interview Results & Scoreboard")
    
    if not st.session_state.workflow_state or st.session_state.workflow_state.get("current_step") != "completed":
        st.info("👆 Please complete the interview session first")
    else:
        state = st.session_state.workflow_state
        
        # Overall performance metrics
        st.markdown("### 🎯 Overall Performance")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            final_score = state.get("final_score", 0)
            st.metric("Final Score", f"{final_score}/10", delta=f"{final_score-5:.1f}")
        
        with col2:
            resume_score = state.get("resume_score", 0)
            st.metric("Resume Score", f"{resume_score:.1f}/10")
        
        with col3:
            st.metric("Experience Level", state.get("experience_level", "N/A"))
        
        with col4:
            if state.get("job_fit"):
                fit_score = state["job_fit"].get("fit_score", 0)
                st.metric("Job Fit Score", f"{fit_score}/10")
        
        with col5:
            questions_answered = len(state.get("answers", []))
            st.metric("Questions Answered", questions_answered)

        # Detailed question-wise feedback
        st.markdown("### 📝 Detailed Question Analysis")
        
        if state.get("answers"):
            for i, answer_data in enumerate(state["answers"]):
                score = answer_data.get("score", 0)
                
                # Color coding based on score
                if score >= 8:
                    score_color = "🟢"
                elif score >= 6:
                    score_color = "🟡"
                else:
                    score_color = "🔴"
                
                with st.expander(f"{score_color} Question {i+1} - Score: {score}/10"):
                    if i < len(st.session_state.current_questions):
                        question = st.session_state.current_questions[i]
                        st.markdown(f"**❓ Question ({question.get('type', 'General')}):**")
                        st.write(question['question'])
                    
                    st.markdown("**✍️ Your Answer:**")
                    st.write(answer_data.get('answer', 'No answer provided'))
                    
                    st.markdown(f"**📊 Score: {score}/10**")
                    st.markdown("**💬 AI Feedback:**")
                    st.info(answer_data.get('feedback', 'No feedback available'))

        # Export functionality
        st.markdown("### 📄 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Summary Report"):
                summary = {
                    "candidate_name": state.get("candidate_name", "Unknown"),
                    "job_title": state.get("job_title", "N/A"),
                    "resume_score": state.get("resume_score", 0),
                    "final_score": state.get("final_score", 0),
                    "experience_level": state.get("experience_level", "N/A"),
                    "job_fit": state.get("job_fit", {}),
                    "total_questions": len(state.get("answers", [])),
                    "evaluation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.json(summary)
        
        with col2:
            # Download detailed results
            detailed_results = {
                "candidate_info": {
                    "name": state.get("candidate_name", "Unknown"),
                    "job_applied": state.get("job_title", "N/A"),
                    "experience_level": state.get("experience_level", "N/A")
                },
                "scores": {
                    "resume_score": state.get("resume_score", 0),
                    "final_score": state.get("final_score", 0),
                    "job_fit_score": state.get("job_fit", {}).get("fit_score", 0)
                },
                "detailed_answers": state.get("answers", []),
                "questions": st.session_state.current_questions,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=json.dumps(detailed_results, indent=2),
                file_name=f"interview_report_{state.get('candidate_name', 'candidate')}.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("*🚀 Built with LangGraph Multi-Agent Architecture | Powered by Google Gemini AI*")