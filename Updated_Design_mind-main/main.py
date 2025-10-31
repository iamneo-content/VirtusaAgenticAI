# main.py - LangGraph-powered HLD Generator with ML Integration
# Streamlit gateway: picks a PRD PDF, runs the LangGraph workflow, and displays results

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import traceback

import streamlit as st
import pandas as pd
import numpy as np

# LangGraph workflow
from workflow import create_hld_workflow
from state.schema import WorkflowInput, ConfigSchema

# UI components
from diagram_publisher import render_mermaid_inline

# ML modules
try:
    from ml.training.generate_dataset import SyntheticDatasetGenerator
    from ml.training.train_large_model import LargeScaleMLTrainer
    from ml.training.inference import HLDQualityPredictor
    from ml.models.quality_scorer import RuleBasedQualityScorer, QualityScore
    ML_AVAILABLE = True
except Exception as e:
    print(f"Warning: ML modules not available: {e}")
    ML_AVAILABLE = False

# ---------- Pretty UI helpers ----------
STYLES = """
<style>
.card{border:1px solid #eee;border-radius:12px;padding:14px;background:#fff;margin:8px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.kv{display:grid;grid-template-columns:120px 1fr;gap:8px;font-size:14px}
.kv b{color:#555}
.pills{margin:6px 0}
.pill{display:inline-block;background:#f1f3f5;border:1px solid #e6e8eb;padding:4px 10px;border-radius:999px;margin:3px;font-size:13px}
.h3{font-weight:600;margin:18px 0 8px}
.smallmuted{font-size:12px;color:#666}
.status-success{color:#28a745;font-weight:600}
.status-processing{color:#ffc107;font-weight:600}
.status-failed{color:#dc3545;font-weight:600}
.status-pending{color:#6c757d;font-weight:600}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

def _to_py(o):
    """Normalize LLM JSON-ish like {'0': 'A', '1': 'B'} -> ['A','B'] recursively."""
    if isinstance(o, dict):
        keys = list(o.keys())
        if keys and all(str(k).isdigit() for k in keys):
            return [_to_py(o[str(i)]) for i in sorted(map(int, keys))]
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_py(v) for v in o]
    return o

def _as_list(x):
    x = _to_py(x)
    if x is None: return []
    return x if isinstance(x, list) else [x]

def _pills(title, items):
    if not items: return
    st.markdown(f"<div class='h3'>{title}</div>", unsafe_allow_html=True)
    st.markdown("<div class='pills'>" + "".join(f"<span class='pill'>{str(i)}</span>" for i in items) + "</div>", unsafe_allow_html=True)

def render_workflow_status(state):
    """Render workflow processing status"""
    if not state.status:
        return

    st.subheader("🔄 Workflow Status")

    status_data = []
    for stage_name, status in state.status.items():
        status_class = f"status-{status.status}"
        status_data.append({
            "Stage": stage_name.replace("_", " ").title(),
            "Status": status.status.title(),
            "Message": status.message or "",
            "Timestamp": status.timestamp.strftime("%H:%M:%S") if status.timestamp else ""
        })

    # Create status DataFrame
    df = pd.DataFrame(status_data)
    st.dataframe(df, width='stretch')

    # Show errors and warnings
    if state.errors:
        st.error("❌ **Errors:**")
        for error in state.errors:
            st.error(f"• {error}")

    if state.warnings:
        st.warning("⚠️ **Warnings:**")
        for warning in state.warnings:
            st.warning(f"• {warning}")

def render_authentication_ui(auth_data):
    """Render authentication analysis results"""
    if not auth_data:
        return

    _pills("Actors", auth_data.actors)
    _pills("Auth Flows", auth_data.flows)
    _pills("Threats", auth_data.threats)
    if auth_data.idp_options:
        _pills("Identity Providers", auth_data.idp_options)

def render_integrations_ui(integrations_data):
    """Render integrations analysis results"""
    if not integrations_data:
        st.info("No integrations found.")
        return

    rows = []
    for integration in integrations_data:
        rows.append({
            "System": integration.system,
            "Purpose": integration.purpose,
            "Protocol": integration.protocol,
            "Auth": integration.auth,
            "Endpoints": ", ".join(integration.endpoints),
            "Inputs": ", ".join(integration.data_contract.get("inputs", [])),
            "Outputs": ", ".join(integration.data_contract.get("outputs", []))
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')

def render_entities_ui(entities_data):
    """Render domain entities"""
    if not entities_data:
        st.info("No entities found.")
        return

    rows = []
    for entity in entities_data:
        rows.append({
            "Entity": entity.name,
            "Attributes Count": len(entity.attributes),
            "Attributes": ", ".join(entity.attributes)
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')

    # Entity details cards
    st.markdown("<div class='h3'>Entity Details</div>", unsafe_allow_html=True)
    st.markdown("<div class='grid'>", unsafe_allow_html=True)
    for entity in entities_data:
        st.markdown(
            "<div class='card'><div style='font-weight:600;margin-bottom:6px'>"
            + entity.name + "</div>"
            + "<div class='pills'>" + "".join(f"<span class='pill'>{a}</span>" for a in entity.attributes) + "</div>"
            + "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def render_apis_ui(apis_data):
    """Render API specifications"""
    if not apis_data:
        st.info("No APIs found.")
        return

    rows = []
    for api in apis_data:
        req_fields = ", ".join(api.request.keys()) if api.request else "—"
        res_fields = ", ".join(api.response.keys()) if api.response else "—"

        rows.append({
            "API": api.name,
            "Description": api.description or "—",
            "Request Fields": req_fields,
            "Response Fields": res_fields
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')

def render_use_cases_ui(use_cases):
    """Render use cases"""
    if not use_cases:
        return

    st.markdown("<div class='h3'>Use Cases</div>", unsafe_allow_html=True)
    for uc in use_cases:
        st.markdown(f"- {uc}")

def render_nfrs_ui(nfrs):
    """Render non-functional requirements"""
    if not nfrs:
        return

    st.markdown("<div class='h3'>Non-Functional Requirements</div>", unsafe_allow_html=True)
    for category, items in nfrs.items():
        if items:
            st.markdown(f"**{category.capitalize()}**")
            for item in items:
                st.markdown(f"- {item}")

def render_risks_ui(risks_data):
    """Render risks and assumptions"""
    if not risks_data:
        return

    rows = []
    for risk in risks_data:
        rows.append({
            "ID": risk.id,
            "Description": risk.desc,
            "Assumption": risk.assumption,
            "Mitigation": risk.mitigation,
            "Impact": risk.impact,
            "Likelihood": risk.likelihood
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')

def render_quality_score_ui(quality_score):
    """Render quality assessment results"""
    if not quality_score:
        return

    st.subheader("📊 Quality Assessment")

    # Overall score with visual
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Overall Score", f"{quality_score.overall_score:.1f}/100")

    with col2:
        st.metric("Completeness", f"{quality_score.completeness:.1f}/100")

    with col3:
        st.metric("Clarity", f"{quality_score.clarity:.1f}/100")

    with col4:
        st.metric("Consistency", f"{quality_score.consistency:.1f}/100")

    with col5:
        st.metric("Security", f"{quality_score.security:.1f}/100")

    # Recommendations
    if quality_score.recommendations:
        st.subheader("💡 Recommendations")
        for rec in quality_score.recommendations:
            st.info(f"• {rec}")

    # Missing elements
    if quality_score.missing_elements:
        st.subheader("❌ Missing Elements")
        for element in quality_score.missing_elements:
            st.warning(f"• {element}")

def list_requirement_pdfs(folder: str = "data") -> List[str]:
    """List available PDF files with detailed information"""
    base = Path(folder)
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return []

    pdf_files = []
    for pdf_path in base.glob("*.pdf"):
        # Only include actual PDF files (not .gitkeep or other files)
        if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
            pdf_files.append(str(pdf_path))

    return sorted(pdf_files)

def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
    """Get information about a PDF file"""
    path = Path(pdf_path)
    if not path.exists():
        return {}

    try:
        stat = path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stat.st_mtime)

        return {
            "name": path.name,
            "size_mb": round(size_mb, 2),
            "modified": modified.strftime("%Y-%m-%d %H:%M"),
            "path": str(path)
        }
    except Exception:
        return {"name": path.name, "path": str(path)}

def render_ml_training_section():
    """Render ML training interface"""
    if not ML_AVAILABLE:
        st.error("ML modules not available. Please install required dependencies.")
        return

    st.header("🤖 ML Model Training")
    st.write("Train machine learning models on a dataset of 30,000 synthetic HLD samples with 38 features.")

    col1, col2 = st.columns(2)

    # Left column: Dataset generation
    with col1:
        st.subheader("📊 Dataset Generation")

        if st.button("🔄 Generate 30,000 Row Dataset", key="generate_dataset"):
            with st.spinner("Generating synthetic dataset..."):
                try:
                    generator = SyntheticDatasetGenerator(random_state=42)
                    df = generator.generate(n_samples=30000)

                    output_path = Path("ml/training/synthetic_hld_dataset.csv")
                    generator.save_dataset(df, str(output_path))

                    st.success("✅ Dataset generated successfully!")

                    # Display statistics
                    st.subheader("Dataset Statistics")
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                    with stats_col1:
                        st.metric("Shape", f"{df.shape[0]} × {df.shape[1]}")

                    with stats_col2:
                        st.metric("Mean Score", f"{df['quality_score'].mean():.2f}")

                    with stats_col3:
                        st.metric("Std Dev", f"{df['quality_score'].std():.2f}")

                    with stats_col4:
                        st.metric("Range", f"{df['quality_score'].min():.0f}-{df['quality_score'].max():.0f}")

                except Exception as e:
                    st.error(f"Error generating dataset: {str(e)}")
                    st.write(traceback.format_exc())

    # Right column: Model training
    with col2:
        st.subheader("🎯 Model Training")

        if st.button("🚀 Train ML Models", key="train_models"):
            with st.spinner("Training models (this may take a few minutes)..."):
                try:
                    trainer = LargeScaleMLTrainer()

                    # Load dataset
                    dataset_path = Path("ml/training/synthetic_hld_dataset.csv")
                    if not dataset_path.exists():
                        st.warning("Dataset not found. Generating dataset first...")
                        generator = SyntheticDatasetGenerator(random_state=42)
                        df = generator.generate(n_samples=30000)
                        generator.save_dataset(df, str(dataset_path))

                    trainer.load_dataset(str(dataset_path))
                    trainer.prepare_data(trainer.df)
                    trainer.train_models()
                    trainer.evaluate_models()
                    trainer.save_models()

                    st.success("✅ Models trained successfully!")

                    # Display results
                    st.subheader("Model Performance")

                    results_data = []
                    for model_name, metrics in trainer.results.items():
                        results_data.append({
                            "Model": model_name,
                            "Test R²": f"{metrics['Test']['R2']:.4f}",
                            "Test RMSE": f"{metrics['Test']['RMSE']:.4f}",
                            "Test MAE": f"{metrics['Test']['MAE']:.4f}",
                            "Test MAPE": f"{metrics['Test']['MAPE']:.4f}%"
                        })

                    st.dataframe(pd.DataFrame(results_data), width='stretch')

                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
                    st.write(traceback.format_exc())

def render_ml_inference_section():
    """Render quality prediction interface"""
    if not ML_AVAILABLE:
        st.error("ML modules not available. Please install required dependencies.")
        return

    st.header("🔮 HLD Quality Prediction")
    st.write("Predict the quality of HLD documents using trained ML models.")

    # Initialize predictor
    predictor = HLDQualityPredictor()

    # Try to load models
    if not predictor.load_models_from_disk():
        st.warning("⚠️ Models not found. Please train models first in the ML Training tab.")
        st.info("Load training data and click 'Train ML Models' to create the models.")
        return

    st.success("✅ Models loaded successfully!")

    # Create tabs for different prediction modes
    tab1, tab2, tab3 = st.tabs(["Quick Scenario", "Custom Features", "Feature Guide"])

    with tab1:
        st.subheader("Quick Scenario Predictions")

        scenario = st.selectbox(
            "Select HLD scenario:",
            ["Excellent HLD", "Good HLD", "Poor HLD"],
            help="Predefined scenarios to test predictions"
        )

        if st.button("📈 Predict Quality Score", key="quick_predict"):
            # Define scenarios
            scenarios = {
                "Excellent HLD": {
                    'word_count': 4500, 'sentence_count': 400, 'avg_sentence_length': 20,
                    'avg_word_length': 5.5, 'header_count': 35, 'code_block_count': 15,
                    'table_count': 10, 'list_count': 25, 'diagram_count': 8,
                    'completeness_score': 95, 'security_mentions': 18, 'scalability_mentions': 17,
                    'api_mentions': 22, 'database_mentions': 14, 'performance_mentions': 16,
                    'monitoring_mentions': 13, 'duplicate_headers': 1, 'header_coverage': 0.95,
                    'code_coverage': 0.7, 'keyword_density': 0.08, 'section_density': 0.7,
                    'has_architecture_section': 1, 'has_security_section': 1,
                    'has_scalability_section': 1, 'has_deployment_section': 1,
                    'has_monitoring_section': 1, 'has_api_spec': 1, 'has_data_model': 1,
                    'service_count': 12, 'entity_count': 35, 'api_endpoint_count': 45,
                    'readability_score': 90, 'completeness_index': 0.95, 'consistency_index': 0.92,
                    'documentation_quality': 92, 'technical_terms_density': 0.25, 'acronym_count': 25
                },
                "Good HLD": {
                    'word_count': 3000, 'sentence_count': 250, 'avg_sentence_length': 16,
                    'avg_word_length': 5.2, 'header_count': 20, 'code_block_count': 8,
                    'table_count': 5, 'list_count': 15, 'diagram_count': 4,
                    'completeness_score': 75, 'security_mentions': 10, 'scalability_mentions': 9,
                    'api_mentions': 12, 'database_mentions': 8, 'performance_mentions': 8,
                    'monitoring_mentions': 6, 'duplicate_headers': 2, 'header_coverage': 0.75,
                    'code_coverage': 0.5, 'keyword_density': 0.06, 'section_density': 0.5,
                    'has_architecture_section': 1, 'has_security_section': 1,
                    'has_scalability_section': 1, 'has_deployment_section': 0,
                    'has_monitoring_section': 0, 'has_api_spec': 1, 'has_data_model': 1,
                    'service_count': 8, 'entity_count': 20, 'api_endpoint_count': 25,
                    'readability_score': 70, 'completeness_index': 0.70, 'consistency_index': 0.75,
                    'documentation_quality': 68, 'technical_terms_density': 0.15, 'acronym_count': 12
                },
                "Poor HLD": {
                    'word_count': 800, 'sentence_count': 80, 'avg_sentence_length': 12,
                    'avg_word_length': 4.5, 'header_count': 8, 'code_block_count': 2,
                    'table_count': 1, 'list_count': 5, 'diagram_count': 0,
                    'completeness_score': 25, 'security_mentions': 2, 'scalability_mentions': 1,
                    'api_mentions': 3, 'database_mentions': 2, 'performance_mentions': 1,
                    'monitoring_mentions': 0, 'duplicate_headers': 5, 'header_coverage': 0.4,
                    'code_coverage': 0.1, 'keyword_density': 0.02, 'section_density': 0.2,
                    'has_architecture_section': 0, 'has_security_section': 0,
                    'has_scalability_section': 0, 'has_deployment_section': 0,
                    'has_monitoring_section': 0, 'has_api_spec': 0, 'has_data_model': 0,
                    'service_count': 2, 'entity_count': 5, 'api_endpoint_count': 5,
                    'readability_score': 30, 'completeness_index': 0.25, 'consistency_index': 0.35,
                    'documentation_quality': 20, 'technical_terms_density': 0.05, 'acronym_count': 3
                }
            }

            features = scenarios[scenario]
            predictions = predictor.predict(features)

            st.subheader(f"Predictions for: {scenario}")

            col1, col2, col3 = st.columns(3)

            for i, (model, score) in enumerate(predictions.items()):
                if model == 'ensemble_average':
                    continue

                if i == 0:
                    with col1:
                        st.metric(model, f"{score:.2f}/100")
                elif i == 1:
                    with col2:
                        st.metric(model, f"{score:.2f}/100")
                else:
                    with col3:
                        st.metric(model, f"{score:.2f}/100")

            if 'ensemble_average' in predictions:
                st.success(f"📊 **Ensemble Average: {predictions['ensemble_average']:.2f}/100**")

    with tab2:
        st.subheader("Custom Feature Input")

        # Create sliders for key features
        st.write("Adjust the following features to see how they affect the quality prediction:")

        col1, col2 = st.columns(2)

        with col1:
            word_count = st.slider("Word Count", 500, 5000, 2000)
            sentence_count = st.slider("Sentence Count", 50, 500, 200)
            header_count = st.slider("Header Count", 5, 40, 15)
            code_block_count = st.slider("Code Blocks", 0, 20, 5)
            completeness_score = st.slider("Completeness Score", 0, 100, 70)
            security_mentions = st.slider("Security Mentions", 0, 20, 8)

        with col2:
            has_architecture = st.checkbox("Has Architecture Section", value=True)
            has_security = st.checkbox("Has Security Section", value=True)
            has_scalability = st.checkbox("Has Scalability Section", value=False)
            has_api_spec = st.checkbox("Has API Spec", value=True)
            has_data_model = st.checkbox("Has Data Model", value=True)
            service_count = st.slider("Service Count", 1, 15, 5)

        if st.button("🔮 Predict with Custom Features", key="custom_predict"):
            # Build feature dictionary
            features = {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': word_count / max(sentence_count, 1),
                'avg_word_length': 5.0,
                'header_count': header_count,
                'code_block_count': code_block_count,
                'table_count': 3,
                'list_count': 10,
                'diagram_count': 2,
                'completeness_score': completeness_score,
                'security_mentions': security_mentions,
                'scalability_mentions': 5,
                'api_mentions': 8,
                'database_mentions': 4,
                'performance_mentions': 4,
                'monitoring_mentions': 2,
                'duplicate_headers': 1,
                'header_coverage': 0.6,
                'code_coverage': 0.3,
                'keyword_density': 0.05,
                'section_density': 0.4,
                'has_architecture_section': int(has_architecture),
                'has_security_section': int(has_security),
                'has_scalability_section': int(has_scalability),
                'has_deployment_section': 0,
                'has_monitoring_section': 0,
                'has_api_spec': int(has_api_spec),
                'has_data_model': int(has_data_model),
                'service_count': service_count,
                'entity_count': 15,
                'api_endpoint_count': 20,
                'readability_score': 65,
                'completeness_index': completeness_score / 100,
                'consistency_index': 0.7,
                'documentation_quality': 60,
                'technical_terms_density': 0.10,
                'acronym_count': 10
            }

            predictions = predictor.predict(features)

            st.subheader("Prediction Results")
            col1, col2, col3 = st.columns(3)

            for i, (model, score) in enumerate(predictions.items()):
                if model == 'ensemble_average':
                    continue

                if i == 0:
                    with col1:
                        st.metric(model, f"{score:.2f}/100")
                elif i == 1:
                    with col2:
                        st.metric(model, f"{score:.2f}/100")
                else:
                    with col3:
                        st.metric(model, f"{score:.2f}/100")

            if 'ensemble_average' in predictions:
                st.success(f"📊 **Ensemble Average: {predictions['ensemble_average']:.2f}/100**")

    with tab3:
        st.subheader("Feature Value Ranges and Guidance")
        predictor.print_feature_guide()


def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="DesignMind GenAI - LangGraph", layout="wide")
    st.title("🧠 DesignMind – LangGraph-Powered Architecture")
    st.caption("AI-driven High-Level Design generation with ML-based quality assessment. Three powerful workflows: Generate HLD, Train ML Models, and Predict Quality.")

    # Create tabs for three main sections
    tab1, tab2, tab3 = st.tabs(["🏗️ HLD Generation", "🤖 ML Training", "🔮 Quality Prediction"])

    # ============ TAB 1: HLD GENERATION ============
    with tab1:
        # Quick overview of available PDFs
        pdf_files = list_requirement_pdfs()
        if pdf_files:
            st.success(f"🎉 **Ready to go!** Found {len(pdf_files)} requirement documents: {', '.join([Path(p).name for p in pdf_files[:3]])}{'...' if len(pdf_files) > 3 else ''}")
        else:
            st.warning("🚨 **No PDF files found!** Please upload requirement documents to the `data/` folder first.")
            st.info("📚 **Expected files:** Requirement-1.pdf, Banking-System-PRD.pdf, E-commerce-Requirements.pdf, etc.")
            st.stop()

        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")

            # Workflow type selection
            workflow_type = st.selectbox(
                "Workflow Type",
                ["sequential", "parallel", "conditional"],
                index=0,
                help="Sequential: One stage at a time (most reliable). Parallel: Optimized sequential execution. Conditional: Smart routing based on state."
            )

            # Diagram configuration
            st.subheader("📊 Diagram Settings")
            render_images = st.checkbox("Generate diagram images", value=True)
            image_format = st.radio("Image format", ["svg", "png"], horizontal=True, index=1)
            renderer = st.radio("Renderer", ["kroki", "mmdc"], horizontal=True, index=0)
            theme = st.selectbox("Diagram theme", ["default", "neutral", "dark"], index=0)

        # Main content
        left, right = st.columns([2, 1])

        with left:
            st.subheader("📄 Select Requirements Document")
            file_names = [Path(p).name for p in pdf_files]
            options = ["— Select a requirements file —"] + file_names
            selected_label = st.selectbox(
                "Choose a PDF document to analyze:",
                options,
                index=0,
                help="Select one of the uploaded PDF requirement documents to generate HLD"
            )

        with right:
            st.subheader("🔧 Configuration")
            st.info(f"**Workflow Mode:** {workflow_type.title()}")

            # PDF Statistics
            st.metric(
                label="📁 Available Documents",
                value=len(pdf_files),
                help="Number of PDF files found in data/ folder"
            )

            # Show PDF file details
            with st.expander("📋 View All PDF Details", expanded=False):
                pdf_data = []
                total_size = 0
                for pdf_path in pdf_files:
                    info = get_pdf_info(pdf_path)
                    if info:
                        size_mb = info.get("size_mb", 0)
                        total_size += size_mb if isinstance(size_mb, (int, float)) else 0
                        pdf_data.append({
                            "File": info["name"],
                            "Size (MB)": size_mb,
                            "Modified": info.get("modified", "N/A")
                        })

                if pdf_data:
                    st.caption(f"Total size: {round(total_size, 2)} MB")
                    df = pd.DataFrame(pdf_data)
                    st.dataframe(df, width='stretch', hide_index=True)

        # Get selected PDF path and show file info
        selected_path = None
        if selected_label != "— Select a requirements file —":
            try:
                selected_index = file_names.index(selected_label)
                selected_path = pdf_files[selected_index]

                # Show selected file information
                if selected_path:
                    info = get_pdf_info(selected_path)
                    if info:
                        st.success(f"📄 **Selected:** {info['name']} ({info.get('size_mb', 'N/A')} MB, modified {info.get('modified', 'N/A')})")
            except ValueError:
                selected_path = None

        # Generate HLD button
        st.divider()

        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            generate_button = st.button(
                "🚀 Generate High-Level Design",
                type="primary",
                disabled=not selected_path,
                width='stretch',
                help="Start the LangGraph workflow to generate comprehensive HLD documentation"
            )

        if generate_button:
            if not selected_path:
                st.warning("Please choose a requirements PDF.")
                st.stop()

            # Create configuration
            config = ConfigSchema(
                render_images=render_images,
                image_format=image_format,
                renderer=renderer,
                theme=theme
            )

            # Create workflow input
            workflow_input = WorkflowInput(
                pdf_path=selected_path,
                config=config
            )

            # Create and run workflow
            workflow = create_hld_workflow(workflow_type)

            # Progress tracking
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            with st.spinner(f"🔄 Running {workflow_type} workflow..."):
                try:
                    # Run workflow
                    result = workflow.run(workflow_input)

                    progress_bar.progress(100)

                    if result.success:
                        status_placeholder.success(f"✅ HLD generated successfully in {result.processing_time:.2f}s")
                        st.balloons()  # Celebration animation

                        # Display results
                        state = result.state

                        # Workflow status
                        render_workflow_status(state)

                        # Extracted requirements
                        if state.extracted:
                            st.header("📋 Extracted Requirements")
                            with st.expander("View extracted content", expanded=False):
                                st.code(state.extracted.markdown[:5000] + "..." if len(state.extracted.markdown) > 5000 else state.extracted.markdown)

                        # Authentication
                        if state.authentication:
                            st.header("🔐 Authentication")
                            render_authentication_ui(state.authentication)

                        # Integrations
                        if state.integrations:
                            st.header("🔗 Integrations")
                            render_integrations_ui(state.integrations)

                        # Domain entities
                        if state.domain and state.domain.entities:
                            st.header("🏗️ Domain Entities")
                            render_entities_ui(state.domain.entities)

                        # APIs
                        if state.domain and state.domain.apis:
                            st.header("🔌 APIs")
                            render_apis_ui(state.domain.apis)

                        # Use cases
                        if state.behavior and state.behavior.use_cases:
                            st.header("📝 Use Cases")
                            render_use_cases_ui(state.behavior.use_cases)

                        # NFRs
                        if state.behavior and state.behavior.nfrs:
                            st.header("⚡ Non-Functional Requirements")
                            render_nfrs_ui(state.behavior.nfrs)

                        # Risks
                        if state.behavior and state.behavior.risks:
                            st.header("⚠️ Risks & Assumptions")
                            render_risks_ui(state.behavior.risks)

                        # Risk heatmap
                        if result.output_paths.get("risk_heatmap"):
                            st.header("🎯 Risk Heatmap")
                            st.image(result.output_paths["risk_heatmap"], caption="Impact × Likelihood (1..5)")

                        # Diagrams
                        if state.diagrams:
                            st.header("📊 Diagrams")

                            # Class diagram
                            if state.diagrams.class_text:
                                st.subheader("🏗️ Class Diagram")
                                render_mermaid_inline(state.diagrams.class_text, key="class", height=560, theme=theme)

                            # Sequence diagrams
                            if state.diagrams.sequence_texts:
                                st.subheader("🔄 Sequence Diagrams")
                                for i, seq_text in enumerate(state.diagrams.sequence_texts, 1):
                                    st.markdown(f"**Sequence #{i}**")
                                    render_mermaid_inline(seq_text, key=f"seq-{i}", height=460, theme=theme)

                        # Download section
                        st.header("💾 Downloads")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if result.output_paths.get("hld_md"):
                                with open(result.output_paths["hld_md"], "rb") as f:
                                    st.download_button(
                                        "📄 Download HLD.md",
                                        data=f,
                                        file_name="HLD.md",
                                        mime="text/markdown"
                                    )

                        with col2:
                            if result.output_paths.get("hld_html"):
                                with open(result.output_paths["hld_html"], "rb") as f:
                                    st.download_button(
                                        "🌐 Download HLD.html",
                                        data=f,
                                        file_name="HLD.html",
                                        mime="text/html"
                                    )

                        with col3:
                            if result.output_paths.get("diagrams_html"):
                                with open(result.output_paths["diagrams_html"], "rb") as f:
                                    st.download_button(
                                        "📊 Download Diagrams.html",
                                        data=f,
                                        file_name="Diagrams.html",
                                        mime="text/html"
                                    )

                        # Output info
                        if state.output:
                            st.info(f"📁 **Output directory:** `{state.output.output_dir}`")

                        st.caption(f"⏱️ Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    else:
                        status_placeholder.error("❌ HLD generation failed")
                        st.error("**Errors:**")
                        for error in result.errors:
                            st.error(f"• {error}")

                        if result.warnings:
                            st.warning("**Warnings:**")
                            for warning in result.warnings:
                                st.warning(f"• {warning}")

                except Exception as e:
                    progress_bar.progress(0)
                    status_placeholder.error(f"❌ Workflow execution failed: {str(e)}")
                    st.exception(e)

    # ============ TAB 2: ML TRAINING ============
    with tab2:
        render_ml_training_section()

    # ============ TAB 3: QUALITY PREDICTION ============
    with tab3:
        render_ml_inference_section()


if __name__ == "__main__":
    main()
