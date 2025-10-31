from typing import Dict, List, TypedDict
import os
import re
import json

class CodeCrafterState(TypedDict):
    """
    State schema for the CodeCrafter LangGraph workflow.
    """
    # Input from user
    user_story: str
    language: str
    
    # Planning results
    features: List[str]
    services: List[str]
    architecture_hints: Dict[str, str]
    architecture_config: Dict[str, str]
    planning_error: str
    
    # Code generation results
    service_outputs: Dict[str, Dict[str, str]]  # {service_name: {filename: content}}
    codegen_error: str
    
    # Swagger results
    swagger_outputs: Dict[str, Dict[str, str]]  # {service_name: {filename: content}}
    swagger_error: str
    
    # Test results
    test_outputs: Dict[str, Dict[str, str]]  # {service_name: {filename: content}}
    test_error: str
    
    # Output directory information
    output_base: str
    output_dir: str
    
    # Status flags
    planning_complete: bool
    codegen_complete: bool
    swagger_complete: bool
    tests_complete: bool
    error_occurred: bool


def slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text[:50].strip("_") or "story"


def save_file(path: str, content: str):
    """Save content to file, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_all_outputs(output_dir: str, service_outputs: Dict[str, Dict[str, str]], 
                      swagger_outputs: Dict[str, Dict[str, str]], 
                      test_outputs: Dict[str, Dict[str, str]]):
    """Write all generated outputs to the file system."""
    # Write service code files
    for service, files in service_outputs.items():
        service_path = os.path.join(output_dir, service)
        save_file(os.path.join(service_path, files.get("controller_filename", f"controller.{get_file_extension(files.get('language', 'python'))}")), 
                 files.get("controller_code", ""))
        save_file(os.path.join(service_path, files.get("service_filename", f"service.{get_file_extension(files.get('language', 'python'))}")), 
                 files.get("service_code", ""))
        save_file(os.path.join(service_path, files.get("model_filename", f"model.{get_file_extension(files.get('language', 'python'))}")), 
                 files.get("model_code", ""))

    # Write swagger documentation
    for service, doc in swagger_outputs.items():
        path = os.path.join(output_dir, "swagger", service, doc.get("filename", "swagger.yaml"))
        save_file(path, doc.get("content", ""))

    # Write test files
    for service, test in test_outputs.items():
        path = os.path.join(output_dir, "tests", service, test.get("filename", f"test.{get_file_extension(test.get('language', 'python'))}"))
        save_file(path, test.get("content", ""))


def get_file_extension(language: str) -> str:
    """Get appropriate file extension for the selected language."""
    ext_map = {
        "Java": "java",
        "NodeJS": "js",
        ".NET": "cs",
        "Python": "py",
        "Go": "go",
        "Ruby": "rb",
        "PHP": "php",
        "Kotlin": "kt"
    }
    return ext_map.get(language, "txt")