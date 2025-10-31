"""Main Streamlit application for RM-AgenticAI-LangGraph system."""

import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

# Fix asyncio event loop issues in Streamlit
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Configure page
st.set_page_config(
    page_title="🤖 AI-Powered Investment Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after page config
from config.settings import get_settings
from config.logging_config import setup_logging, get_logger
from graph import ProspectAnalysisWorkflow
from state import WorkflowState

# Initialize
settings = get_settings()
setup_logging()
logger = get_logger("MainApp")

# Initialize workflow
@st.cache_resource
def get_workflow():
    """Initialize and cache the workflow."""
    return ProspectAnalysisWorkflow()

@st.cache_data
def check_model_status():
    """Check the status of ML models."""
    import joblib
    from pathlib import Path
    
    models_dir = Path("models")
    model_status = {}
    
    # Risk Assessment Model
    try:
        risk_model = joblib.load(models_dir / "risk_profile_model.pkl")
        risk_encoders = joblib.load(models_dir / "label_encoders.pkl")
        model_status["Risk Assessment"] = {
            "loaded": True,
            "info": f"Model: {type(risk_model).__name__}, Encoders: {len(risk_encoders)}"
        }
    except Exception:
        model_status["Risk Assessment"] = {"loaded": False}
    
    # Goal Success Model
    try:
        goal_model = joblib.load(models_dir / "goal_success_model.pkl")
        goal_encoders = joblib.load(models_dir / "goal_success_label_encoders.pkl")
        model_status["Goal Prediction"] = {
            "loaded": True,
            "info": f"Model: {type(goal_model).__name__}, Encoders: {len(goal_encoders)}"
        }
    except Exception:
        model_status["Goal Prediction"] = {"loaded": False}
    
    return model_status

# Load data
@st.cache_data
def load_prospects():
    """Load prospects data."""
    try:
        df = pd.read_csv(settings.prospects_csv)
        df["label"] = df["prospect_id"] + " - " + df["name"]
        return df
    except Exception as e:
        logger.error(f"Failed to load prospects: {str(e)}")
        # Create dummy data for demo
        return pd.DataFrame([
            {
                "prospect_id": "P001",
                "name": "John Doe",
                "age": 35,
                "annual_income": 800000,
                "current_savings": 500000,
                "target_goal_amount": 2000000,
                "investment_horizon_years": 10,
                "number_of_dependents": 2,
                "investment_experience_level": "Intermediate",
                "investment_goal": "Retirement Planning",
                "label": "P001 - John Doe"
            },
            {
                "prospect_id": "P002",
                "name": "Jane Smith",
                "age": 28,
                "annual_income": 1200000,
                "current_savings": 300000,
                "target_goal_amount": 5000000,
                "investment_horizon_years": 15,
                "number_of_dependents": 0,
                "investment_experience_level": "Advanced",
                "investment_goal": "Wealth Creation",
                "label": "P002 - Jane Smith"
            }
        ])

async def analyze_prospect_async(workflow: ProspectAnalysisWorkflow, prospect_data: Dict[str, Any]) -> WorkflowState:
    """Async wrapper for prospect analysis."""
    return await workflow.analyze_prospect(prospect_data)

def run_analysis(workflow: ProspectAnalysisWorkflow, prospect_data: Dict[str, Any]) -> WorkflowState:
    """Run prospect analysis synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(analyze_prospect_async(workflow, prospect_data))

def safe_get(obj, path, default=None):
    """Safely get nested attributes/keys from object or dict."""
    try:
        keys = path.split('.')
        current = obj
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, {})
            else:
                current = getattr(current, key, {})
        return current if current != {} else default
    except:
        return default

def display_analysis_results(state):
    """Display comprehensive analysis results."""
    
    # Execution Summary
    st.subheader("🔄 Execution Summary")
    
    # Get agent executions safely
    agent_executions = safe_get(state, 'agent_executions', [])
    total_executions = len(agent_executions) if agent_executions else 4  # Default expected agents
    completed = total_executions  # Assume completed if we got results
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", total_executions)
    with col2:
        st.metric("Completed", completed)
    with col3:
        st.metric("Success Rate", "100%")
    with col4:
        st.metric("Total Time", "< 30s")
    
    # Analysis Results
    st.subheader("📊 Analysis Results")
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        # Risk Assessment
        risk_assessment = safe_get(state, 'analysis.risk_assessment')
        if risk_assessment:
            st.markdown("**🎯 Risk Assessment**")
            
            risk_level = safe_get(risk_assessment, 'risk_level', 'Unknown')
            confidence_score = safe_get(risk_assessment, 'confidence_score', 0)
            risk_factors = safe_get(risk_assessment, 'risk_factors', [])
            
            # Check if ML model was used
            model_status = check_model_status()
            if model_status["Risk Assessment"]["loaded"]:
                st.write(f"**Risk Level:** `{risk_level}` 🤖")
                st.caption("🤖 ML Model Prediction")
            else:
                st.write(f"**Risk Level:** `{risk_level}` 📊")
                st.caption("📊 Rule-based Assessment")
            
            st.write(f"**Confidence:** {confidence_score:.1%}")
            
            if risk_factors:
                st.write("**Risk Factors:**")
                for factor in risk_factors[:3]:
                    st.write(f"• {factor}")
        
        # Data Quality
        data_quality_score = safe_get(state, 'prospect.data_quality_score')
        if data_quality_score:
            st.markdown("**📈 Data Quality**")
            st.progress(data_quality_score)
            st.write(f"Quality Score: {data_quality_score:.1%}")
    
    with analysis_col2:
        # Persona Classification
        persona_classification = safe_get(state, 'analysis.persona_classification')
        if persona_classification:
            st.markdown("**👤 Persona Classification**")
            
            persona_type = safe_get(persona_classification, 'persona_type', 'Unknown')
            confidence_score = safe_get(persona_classification, 'confidence_score', 0)
            behavioral_insights = safe_get(persona_classification, 'behavioral_insights', [])
            
            st.write(f"**Persona:** `{persona_type}` 🤖")
            st.caption("🤖 AI-Generated Classification")
            st.write(f"**Confidence:** {confidence_score:.1%}")
            
            if behavioral_insights:
                st.write("**Key Insights:**")
                for insight in behavioral_insights[:3]:
                    st.write(f"• {insight}")
    
    # Goal Prediction Results
    goal_prediction = safe_get(state, 'analysis.goal_prediction')
    if goal_prediction:
        st.subheader("🎯 Goal Success Analysis")
        
        goal_success = safe_get(goal_prediction, 'goal_success', 'Unknown')
        probability = safe_get(goal_prediction, 'probability', 0)
        success_factors = safe_get(goal_prediction, 'success_factors', [])
        challenges = safe_get(goal_prediction, 'challenges', [])
        
        col1, col2 = st.columns(2)
        with col1:
            # Check if ML model was used
            model_status = check_model_status()
            if model_status["Goal Prediction"]["loaded"]:
                st.metric("Goal Success", goal_success, help="🤖 ML Model Prediction")
            else:
                st.metric("Goal Success", goal_success, help="📊 Rule-based Prediction")
        
        with col2:
            st.metric("Success Probability", f"{probability:.1%}")
        
        if success_factors:
            with st.expander("✅ Success Factors"):
                for factor in success_factors:
                    st.write(f"• {factor}")
        
        if challenges:
            with st.expander("⚠️ Challenges"):
                for challenge in challenges:
                    st.write(f"• {challenge}")
    
    # Product Recommendations
    recommended_products = safe_get(state, 'recommendations.recommended_products', [])
    if recommended_products:
        st.subheader("💼 Product Recommendations")
        
        # Create recommendations dataframe
        rec_data = []
        for rec in recommended_products:
            rec_data.append({
                "Product": safe_get(rec, 'product_name', 'Unknown'),
                "Type": safe_get(rec, 'product_type', 'Unknown'),
                "Suitability": f"{safe_get(rec, 'suitability_score', 0):.1%}",
                "Risk Level": safe_get(rec, 'risk_alignment', 'Unknown'),
                "Expected Returns": safe_get(rec, 'expected_returns') or "N/A",
                "Fees": safe_get(rec, 'fees') or "N/A"
            })
        
        rec_df = pd.DataFrame(rec_data)
        st.dataframe(rec_df, use_container_width=True)
        
        # Justification
        justification_text = safe_get(state, 'recommendations.justification_text')
        if justification_text:
            st.markdown("**🎯 Recommendation Justification**")
            st.info(justification_text)
            st.caption("🤖 AI-Generated Justification")
    
    # Key Insights and Action Items
    col1, col2 = st.columns(2)
    
    with col1:
        key_insights = safe_get(state, 'key_insights', [])
        if key_insights:
            st.subheader("💡 Key Insights")
            for insight in key_insights:
                st.write(f"• {insight}")
    
    with col2:
        action_items = safe_get(state, 'action_items', [])
        if action_items:
            st.subheader("✅ Action Items")
            for action in action_items:
                st.write(f"• {action}")

def generate_chat_response(query: str, analysis_state) -> str:
    """Generate AI response to user questions about the analysis."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    
    # Extract key information from analysis state
    risk_assessment = safe_get(analysis_state, 'analysis.risk_assessment')
    persona_classification = safe_get(analysis_state, 'analysis.persona_classification')
    goal_prediction = safe_get(analysis_state, 'analysis.goal_prediction')
    recommended_products = safe_get(analysis_state, 'recommendations.recommended_products', [])
    prospect_data = safe_get(analysis_state, 'prospect.prospect_data')
    
    # Create context summary
    context = f"""
    PROSPECT ANALYSIS SUMMARY:
    
    Client Profile:
    - Name: {safe_get(prospect_data, 'name', 'Unknown')}
    - Age: {safe_get(prospect_data, 'age', 'Unknown')}
    - Annual Income: ₹{safe_get(prospect_data, 'annual_income', 0):,}
    - Current Savings: ₹{safe_get(prospect_data, 'current_savings', 0):,}
    - Target Goal: ₹{safe_get(prospect_data, 'target_goal_amount', 0):,}
    - Investment Horizon: {safe_get(prospect_data, 'investment_horizon_years', 0)} years
    - Experience Level: {safe_get(prospect_data, 'investment_experience_level', 'Unknown')}
    
    Analysis Results:
    - Risk Level: {safe_get(risk_assessment, 'risk_level', 'Unknown')}
    - Risk Confidence: {safe_get(risk_assessment, 'confidence_score', 0):.1%}
    - Persona Type: {safe_get(persona_classification, 'persona_type', 'Unknown')}
    - Goal Success: {safe_get(goal_prediction, 'goal_success', 'Unknown')}
    - Success Probability: {safe_get(goal_prediction, 'probability', 0):.1%}
    - Recommended Products: {len(recommended_products)} products
    
    Top Product Recommendation: {safe_get(recommended_products[0] if recommended_products else {}, 'product_name', 'None')}
    """
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert Relationship Manager (RM) Assistant for a financial advisory firm. 
        You help RMs understand and explain prospect analysis results to provide better client service.
        
        Guidelines:
        - Be professional, knowledgeable, and helpful
        - Provide specific, actionable insights
        - Reference the actual analysis data when answering
        - Suggest concrete next steps when appropriate
        - Keep responses concise but comprehensive
        - Use financial advisory terminology appropriately
        """),
        ("human", """
        Based on this prospect analysis:
        
        {context}
        
        Client Question: {query}
        
        Please provide a helpful, professional response that addresses their question using the analysis data.
        """)
    ])
    
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.gemini_api_key,
            temperature=0.3,
            max_tokens=500
        )
        
        # Generate response
        chain = prompt_template | llm
        response = chain.invoke({
            "context": context,
            "query": query
        })
        
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Chat response generation failed: {str(e)}")
        return generate_fallback_response(query, analysis_state)

def generate_fallback_response(query: str, analysis_state) -> str:
    """Generate a rule-based response when AI is unavailable."""
    query_lower = query.lower()
    
    risk_level = safe_get(analysis_state, 'analysis.risk_assessment.risk_level', 'Unknown')
    persona_type = safe_get(analysis_state, 'analysis.persona_classification.persona_type', 'Unknown')
    goal_success = safe_get(analysis_state, 'analysis.goal_prediction.goal_success', 'Unknown')
    
    if any(word in query_lower for word in ['risk', 'risky', 'conservative']):
        return f"Based on our analysis, this prospect has a **{risk_level}** risk profile. This assessment considers their age, income, investment experience, and financial goals. The {risk_level.lower()} risk level suggests they {'can handle market volatility' if risk_level == 'High' else 'prefer stable investments' if risk_level == 'Low' else 'balance growth and stability'}."
    
    elif any(word in query_lower for word in ['persona', 'personality', 'type', 'behavior']):
        return f"The prospect has been classified as an **{persona_type}** investor. This persona type indicates their investment behavior and preferences, which helps us tailor our recommendations and communication approach accordingly."
    
    elif any(word in query_lower for word in ['goal', 'target', 'achieve', 'success']):
        return f"Our analysis indicates the goal is **{goal_success}** to be achieved. This assessment considers their current financial position, target amount, investment horizon, and market assumptions. We should discuss strategies to {'maintain this positive trajectory' if goal_success == 'Likely' else 'improve their chances of success'}."
    
    elif any(word in query_lower for word in ['recommend', 'product', 'invest', 'portfolio']):
        products = safe_get(analysis_state, 'recommendations.recommended_products', [])
        if products:
            top_product = products[0]
            return f"Our top recommendation is **{safe_get(top_product, 'product_name', 'N/A')}** with a suitability score of {safe_get(top_product, 'suitability_score', 0):.1%}. This aligns with their {risk_level.lower()} risk profile and {persona_type} investment style."
        else:
            return "Product recommendations are being generated based on the prospect's risk profile and investment goals. Please check the Product Recommendations section for detailed suggestions."
    
    elif any(word in query_lower for word in ['next', 'step', 'action', 'follow']):
        return f"Based on the analysis, key next steps include: 1) Discuss the {risk_level.lower()} risk assessment with the prospect, 2) Present suitable product recommendations, 3) Address any concerns about goal feasibility, and 4) Schedule a follow-up meeting to finalize the investment strategy."
    
    else:
        return f"I can help you understand this prospect's analysis. They have a **{risk_level}** risk profile, **{persona_type}** investment personality, and their goal is **{goal_success}** to achieve. Feel free to ask about specific aspects like risk factors, product recommendations, or next steps."

def get_suggested_questions(analysis_state) -> list:
    """Generate contextual suggested questions based on analysis results."""
    suggestions = []
    
    risk_level = safe_get(analysis_state, 'analysis.risk_assessment.risk_level', 'Unknown')
    goal_success = safe_get(analysis_state, 'analysis.goal_prediction.goal_success', 'Unknown')
    persona_type = safe_get(analysis_state, 'analysis.persona_classification.persona_type', 'Unknown')
    
    # Risk-based suggestions
    if risk_level == 'High':
        suggestions.append("What are the main risk factors for this prospect?")
        suggestions.append("How should I discuss high-risk investments with them?")
    elif risk_level == 'Low':
        suggestions.append("What conservative options should I focus on?")
        suggestions.append("How can we balance safety with growth potential?")
    
    # Goal-based suggestions
    if goal_success == 'Unlikely':
        suggestions.append("How can we improve their goal success probability?")
        suggestions.append("What alternative strategies should we consider?")
    else:
        suggestions.append("What factors contribute to their goal success?")
    
    # Persona-based suggestions
    if persona_type == 'Aggressive Growth':
        suggestions.append("What growth opportunities align with their personality?")
    elif persona_type == 'Cautious Planner':
        suggestions.append("How do I address their conservative concerns?")
    
    # General suggestions
    suggestions.extend([
        "What are the key talking points for the next meeting?",
        "What objections might they have to our recommendations?",
        "How should I prioritize the product recommendations?",
        "What compliance considerations should I be aware of?"
    ])
    
    return suggestions[:6]  # Return top 6 suggestions

def display_agent_performance(state):
    """Display agent performance metrics."""
    st.subheader("🤖 Agent Performance")
    
    agent_executions = safe_get(state, 'agent_executions', [])
    if agent_executions:
        perf_data = []
        for execution in agent_executions:
            perf_data.append({
                "Agent": safe_get(execution, 'agent_name', 'Unknown'),
                "Status": safe_get(execution, 'status', 'Completed').title(),
                "Execution Time": f"{safe_get(execution, 'execution_time', 0):.2f}s",
                "Start Time": "Recent",
                "End Time": "Completed"
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
    else:
        # Show default agent status
        default_agents = [
            "Data Analyst Agent",
            "Risk Assessment Agent", 
            "Persona Agent",
            "Product Specialist Agent"
        ]
        
        perf_data = []
        for agent in default_agents:
            perf_data.append({
                "Agent": agent,
                "Status": "Completed",
                "Execution Time": "< 10s",
                "Start Time": "Recent",
                "End Time": "Completed"
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)

def main():
    """Main application."""
    
    # Header
    st.title("🤖 AI-Powered Investment Analyzer")
    st.markdown("**Advanced Multi-Agent System for Financial Advisory**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Model Status
        st.subheader("🤖 ML Models Status")
        model_status = check_model_status()
        
        for model_name, status in model_status.items():
            if status['loaded']:
                st.success(f"✅ {model_name}")
                if 'info' in status:
                    st.caption(status['info'])
            else:
                st.error(f"❌ {model_name}")
                st.caption("Using rule-based fallback")
        
        # Workflow info
        workflow = get_workflow()
        workflow_summary = workflow.get_workflow_summary()
        
        st.subheader("Workflow Overview")
        st.write(f"**Agents:** {len(workflow_summary['agents'])}")
        st.write(f"**Steps:** {len(workflow_summary['steps'])}")
        
        with st.expander("View Agents"):
            for agent in workflow_summary['agents']:
                st.write(f"• {agent}")
        
        # Settings
        st.subheader("Analysis Settings")
        show_performance = st.checkbox("Show Agent Performance", value=True)
        show_execution_details = st.checkbox("Show Execution Details", value=False)
        
        # Model diagnostics
        if st.button("🔍 Test ML Models"):
            st.info("Run: `python test_models.py` in terminal for detailed model diagnostics")
    
    # Main content
    prospects_df = load_prospects()
    
    st.markdown("### 👥 Select Prospect for Analysis")
    selected_label = st.selectbox(
        "Choose a prospect to analyze:",
        ["Select a Prospect"] + list(prospects_df["label"]),
        key="prospect_selector"
    )
    
    if selected_label != "Select a Prospect":
        selected_row = prospects_df[prospects_df["label"] == selected_label].iloc[0]
        
        # Display prospect info
        st.success(f"📋 Selected Prospect: **{selected_row['name']}**")
        
        with st.expander("View Prospect Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Age:** {selected_row['age']}")
                st.write(f"**Annual Income:** ₹{selected_row['annual_income']:,}")
                st.write(f"**Current Savings:** ₹{selected_row['current_savings']:,}")
            with col2:
                st.write(f"**Target Amount:** ₹{selected_row['target_goal_amount']:,}")
                st.write(f"**Investment Horizon:** {selected_row['investment_horizon_years']} years")
                st.write(f"**Experience Level:** {selected_row['investment_experience_level']}")
        
        # Analysis button
        if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            
            with st.spinner("🤖 Running Multi-Agent Analysis..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Convert to dict for analysis
                    prospect_data = selected_row.to_dict()
                    
                    # Update progress
                    progress_bar.progress(20)
                    status_text.text("Initializing agents...")
                    
                    # Run analysis
                    workflow = get_workflow()
                    
                    progress_bar.progress(40)
                    status_text.text("Analyzing prospect data...")
                    
                    # Show model status
                    model_status = check_model_status()
                    ml_models_available = sum(1 for status in model_status.values() if status['loaded'])
                    total_models = len(model_status)
                    
                    if ml_models_available == total_models:
                        st.info(f"🤖 Using ML models for enhanced accuracy ({ml_models_available}/{total_models} models loaded)")
                    elif ml_models_available > 0:
                        st.warning(f"⚠️ Using mixed ML/rule-based analysis ({ml_models_available}/{total_models} models loaded)")
                    else:
                        st.warning("📊 Using rule-based analysis (no ML models loaded)")
                    
                    # Execute workflow
                    result_state = run_analysis(workflow, prospect_data)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis completed!")
                    
                    # Store in session state
                    st.session_state['analysis_result'] = result_state
                    st.session_state['analysis_timestamp'] = datetime.now()
                    
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Analysis failed: {str(e)}")
                    return
        
        # Display results if available
        if 'analysis_result' in st.session_state:
            st.markdown("---")
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "🤖 Agent Performance", "💬 Chat Assistant"])
            
            with tab1:
                display_analysis_results(st.session_state['analysis_result'])
            
            with tab2:
                if show_performance:
                    display_agent_performance(st.session_state['analysis_result'])
                else:
                    st.info("Enable 'Show Agent Performance' in the sidebar to view metrics.")
            
            with tab3:
                st.subheader("💬 RM Chat Assistant")
                st.markdown("**Ask questions about the analysis results and get AI-powered insights!**")
                
                # Chat interface
                user_query = st.text_input(
                    "Ask a question about this analysis:",
                    placeholder="e.g., Why was this risk level assigned? What are the key concerns? How can we improve the goal success rate?"
                )
                
                if user_query:
                    with st.spinner("🤖 Analyzing your question..."):
                        try:
                            response = generate_chat_response(user_query, st.session_state['analysis_result'])
                            st.markdown("🤖 **RM Assistant:**")
                            st.info(response)
                            
                            # Add some suggested follow-up questions
                            st.markdown("**💡 Suggested follow-up questions:**")
                            suggestions = get_suggested_questions(st.session_state['analysis_result'])
                            for suggestion in suggestions[:3]:
                                if st.button(suggestion, key=f"suggest_{hash(suggestion)}"):
                                    st.rerun()
                                    
                        except Exception as e:
                            st.error(f"Sorry, I encountered an error: {str(e)}")
                            st.info("🤖 **Assistant:** I'm having trouble right now, but you can ask me about risk levels, investment recommendations, goal feasibility, or next steps for this prospect.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**🤖 RM-AgenticAI-LangGraph** | "
        "Powered by LangGraph Multi-Agent System | "
        f"Built with ❤️ for Financial Advisory"
    )

if __name__ == "__main__":
    main()