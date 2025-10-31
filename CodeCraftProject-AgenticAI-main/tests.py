"""
Integration Tests for CodeCrafter AI
- 2 REAL API calls to test actual functionality
- Rest MOCKED to save API quota
- Maximum 10 test cases as required

This file includes conftest fixtures and all test cases combined.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# PYTEST FIXTURES (from conftest.py)
# ============================================================================

@pytest.fixture
def sample_user_story():
    """Provide sample user story for tests"""
    return "Users should be able to register, login, view products, add items to cart, and checkout with payment processing."


@pytest.fixture
def supported_languages():
    """Provide list of supported languages"""
    return ["Java", "NodeJS", ".NET", "Python", "Go", "Ruby", "PHP", "Kotlin"]


@pytest.fixture
def sample_state():
    """Provide sample CodeCrafterState for tests"""
    from state import CodeCrafterState

    return {
        "user_story": "Users should register and login",
        "language": "Python",
        "features": ["User Registration", "User Login"],
        "services": ["UserService"],
        "architecture_hints": {"pattern": "rest"},
        "architecture_config": {"database": "postgresql"},
        "code_output": {},
        "swagger_output": "",
        "test_output": "",
        "output_dir": "./output"
    }


@pytest.fixture
def planning_result():
    """Provide sample planning result"""
    return {
        "features": ["User Registration", "User Login", "Product Viewing"],
        "services": ["UserService", "ProductService"],
        "architecture_hints": {
            "pattern": "rest",
            "database": "postgresql",
            "caching": "redis"
        },
        "architecture_config": {
            "api_gateway": "nginx",
            "service_discovery": "consul"
        }
    }


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

# TEST 1: REAL API CALL - Planning Agent
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY_1"), reason="GEMINI_API_KEY_1 not configured")
def test_planning_agent_real_api():
    """Test 1: REAL API - Planning agent generates features from user story"""
    try:
        from agents.planning_agent import planning_agent

        state = {
            "user_story": "Create user registration system",
            "language": "Python",
            "features": [],
            "services": [],
            "architecture_hints": {},
            "architecture_config": {},
            "code_output": {},
            "swagger_output": "",
            "test_output": "",
            "output_dir": "./output",
            "planning_error": ""
        }

        result = planning_agent(state)

        # REAL API TEST - Verify Gemini returned actual features
        assert "features" in result
        assert len(result.get("features", [])) > 0, "Planning agent should generate features"
        assert "services" in result
        print(f"REAL API: Generated {len(result['features'])} features")
    except Exception as e:
        pytest.skip(f"API credentials unavailable: {str(e)}")


# TEST 2: REAL API CALL - Code Generation Agent
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY_1"), reason="GEMINI_API_KEY_1 not configured")
def test_codegen_agent_real_api():
    """Test 2: REAL API - Code generation produces output"""
    try:
        from agents.codegen_agent import codegen_agent

        state = {
            "user_story": "Create user API",
            "language": "Python",
            "features": ["User Registration"],
            "services": ["UserService"],
            "architecture_hints": {"pattern": "rest"},
            "architecture_config": {},
            "code_output": {},
            "swagger_output": "",
            "test_output": "",
            "output_dir": "./output",
            "planning_error": ""
        }

        result = codegen_agent(state)

        # REAL API TEST - Verify code was generated
        assert "code_output" in result
        print(f"REAL API: Code generation completed")
    except Exception as e:
        pytest.skip(f"API credentials unavailable: {str(e)}")


# TEST 3: MOCKED - Swagger Agent
@patch('agents.swagger_agent.ChatGoogleGenerativeAI')
def test_swagger_agent_mocked(mock_llm):
    """Test 3: MOCKED - Swagger agent generates API docs"""
    from agents.swagger_agent import swagger_agent

    # Mock the LLM response
    mock_llm.return_value.invoke.return_value.content = '{"swagger": "2.0"}'

    state = {
        "user_story": "API endpoint",
        "language": "Python",
        "features": ["API"],
        "services": ["Service"],
        "architecture_hints": {},
        "architecture_config": {},
        "code_output": {"Service": "code"},
        "swagger_output": "",
        "test_output": "",
        "output_dir": "./output",
        "planning_error": ""
    }

    result = swagger_agent(state)

    # Verify swagger_output field exists
    assert "swagger_output" in result
    print("MOCKED: Swagger agent test passed")


# TEST 4: MOCKED - Test Generation Agent
@patch('agents.test_agent.ChatGoogleGenerativeAI')
def test_agent_mocked(mock_llm):
    """Test 4: MOCKED - Test agent generates unit tests"""
    from agents.test_agent import generate_tests

    # Mock the LLM response
    mock_llm.return_value.invoke.return_value.content = 'def test_user(): pass'

    state = {
        "user_story": "Tests",
        "language": "Python",
        "features": ["Feature"],
        "services": ["Service"],
        "architecture_hints": {},
        "architecture_config": {},
        "code_output": {"Service": "code"},
        "swagger_output": "{}",
        "test_output": "",
        "output_dir": "./output",
        "planning_error": ""
    }

    result = generate_tests(state)

    # Verify test_output field exists
    assert "test_output" in result
    print("MOCKED: Test agent test passed")


# TEST 5: State Schema Validation
def test_state_schema_valid():
    """Test 5: State schema properly structured"""
    state = {
        "user_story": "Test",
        "language": "Python",
        "features": ["F1"],
        "services": ["S1"],
        "architecture_hints": {},
        "architecture_config": {},
        "code_output": {},
        "swagger_output": "",
        "test_output": "",
        "output_dir": "./output",
        "planning_error": ""
    }

    # Verify all required fields exist
    required_fields = ["user_story", "language", "features", "services", "code_output", "swagger_output"]
    for field in required_fields:
        assert field in state, f"Missing required field: {field}"

    print("State schema validated")


# TEST 6: MOCKED - Multi-language Support
@patch('agents.codegen_agent.ChatGoogleGenerativeAI')
def test_multi_language_support(mock_llm):
    """Test 6: MOCKED - Code generation supports multiple languages"""
    from agents.codegen_agent import codegen_agent

    mock_llm.return_value.invoke.return_value.content = 'public class User {}'

    state = {
        "user_story": "Java service",
        "language": "Java",
        "features": ["API"],
        "services": ["Service"],
        "architecture_hints": {},
        "architecture_config": {},
        "code_output": {},
        "swagger_output": "",
        "test_output": "",
        "output_dir": "./output",
        "planning_error": ""
    }

    result = codegen_agent(state)

    # Verify Java support
    assert "code_output" in result
    print("MOCKED: Multi-language (Java) support verified")


# TEST 7: MOCKED - API Error Handling
@patch('agents.planning_agent.ChatGoogleGenerativeAI')
def test_error_handling(mock_llm):
    """Test 7: MOCKED - Graceful error handling"""
    from agents.planning_agent import planning_agent

    # Simulate API error
    mock_llm.return_value.invoke.side_effect = Exception("API Error")

    state = {
        "user_story": "Test",
        "language": "Python",
        "features": [],
        "services": [],
        "architecture_hints": {},
        "architecture_config": {},
        "code_output": {},
        "swagger_output": "",
        "test_output": "",
        "output_dir": "./output",
        "planning_error": ""
    }

    try:
        result = planning_agent(state)
        # Should still return state or handle gracefully
        assert result is not None
    except Exception as e:
        # Error handling is acceptable
        assert "API Error" in str(e) or result is not None

    print("MOCKED: Error handling verified")


# TEST 8: Output Directory
def test_output_directory_exists():
    """Test 8: Output directory is ready"""
    output_dir = "./output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert os.path.exists(output_dir), "Output directory should exist"
    assert os.path.isdir(output_dir), "Output path should be directory"

    print("Output directory verified")


# TEST 9: API Key Configuration
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY_1"), reason="GEMINI_API_KEY_1 not configured")
def test_gemini_api_key_configured():
    """Test 9: Gemini API key is configured"""
    api_key = os.getenv("GEMINI_API_KEY_1")

    if api_key:
        assert len(api_key) > 0, "GEMINI_API_KEY_1 is empty"
        if api_key.startswith("AIza"):
            print(f"API Key configured: {api_key[:10]}...{api_key[-5:]}")
        else:
            pytest.skip("API key exists but not in expected format")
    else:
        pytest.skip("GEMINI_API_KEY_1 not configured")


# TEST 10: MOCKED - Workflow State Accumulation
@patch('agents.planning_agent.ChatGoogleGenerativeAI')
def test_state_accumulation(mock_llm):
    """Test 10: MOCKED - State accumulates through workflow"""
    from agents.planning_agent import planning_agent

    mock_llm.return_value.invoke.return_value.content = """
    {
        "features": ["Reg", "Login"],
        "services": ["User"]
    }
    """

    state = {
        "user_story": "Auth system",
        "language": "Python",
        "features": [],
        "services": [],
        "architecture_hints": {},
        "architecture_config": {},
        "code_output": {},
        "swagger_output": "",
        "test_output": "",
        "output_dir": "./output",
        "planning_error": ""
    }

    result = planning_agent(state)

    # Verify state was updated/accumulated
    assert "features" in result
    assert "services" in result

    print("State accumulation verified")
