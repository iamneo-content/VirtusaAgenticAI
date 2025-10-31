import os
import re
import json
import streamlit as st
from graph import run_all_agents

def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text[:50].strip("_") or "story"

# === STREAMLIT UI ===
st.set_page_config(page_title="GenAI Microservice Code Generator", layout="wide")
st.title("CodeCrafter AI: GenAI-Powered Microservice Builder")

with st.form("input_form"):
    user_story = st.text_area("Enter User Story", height=150, placeholder="e.g., Users should register, login, subscribe...")
    language = st.selectbox("Choose Backend Language", [
        "Java", "NodeJS", ".NET", "Python", "Go", "Ruby", "PHP", "Kotlin"
    ])
    submit = st.form_submit_button("Generate Microservices")

if submit and user_story.strip():
    with st.spinner("Running all agents..."):
        result = run_all_agents(user_story, language)
        folder_name = os.path.relpath(result.get("output_dir", "output"), start=".")
        st.success("Code generation complete.")
        st.info(f"Files saved to: `{folder_name}`")

        st.subheader("Features & Services")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Features")
            for i, feat in enumerate(result.get("features", [])):
                st.markdown(f"- **{i+1}.** {feat}")
        with col2:
            st.markdown("#### Services")
            for i, svc in enumerate(result.get("services", [])):
                st.markdown(f"- **{i+1}.** `{svc}`")

        arch_hints = result.get("architecture_hints", {})
        non_empty_hints = {k: v for k, v in arch_hints.items() if v.lower() != "none"}
        if non_empty_hints:
            st.markdown("#### User-Specified Architecture Hints")
            for k, v in non_empty_hints.items():
                st.markdown(f"- **{k.replace('_', ' ').title()}**: `{v}`")

        st.markdown("#### Architecture Configuration")
        arch = result.get("architecture_config", {})
        st.markdown(f"""
        - **Architecture**: `{arch.get('architecture')}`
        - **Database**: `{arch.get('database')}`
        - **Messaging**: `{arch.get('messaging')}`
        - **Cache**: `{arch.get('cache')}`
        - **API Gateway**: `{arch.get('api_gateway')}`
        - **Service Discovery**: `{arch.get('service_discovery')}`
        """)

        st.subheader("Generated Code Preview")
        for service, files in result.get("service_outputs", {}).items():
            with st.expander(f"{service}"):
                st.code(files.get("controller_code", ""), language=language.lower())
                st.code(files.get("service_code", ""), language=language.lower())
                st.code(files.get("model_code", ""), language=language.lower())

        st.subheader("Swagger YAMLs")
        for service, doc in result.get("swagger_outputs", {}).items():
            with st.expander(f"{service} Swagger"):
                st.code(doc.get("content", ""), language="yaml")

        st.subheader("Unit Tests")
        for service, test in result.get("test_outputs", {}).items():
            with st.expander(f"{service} Test"):
                st.code(test.get("content", ""), language=language.lower())
