# Deprecated: This file has been moved to graph.py
# Import from the new location for backwards compatibility

from graph import (
    app,
    evaluation_app,
    create_workflow,
    create_evaluation_workflow,
)

__all__ = ["app", "evaluation_app", "create_workflow", "create_evaluation_workflow"]

# Note: This file is kept for backwards compatibility.
# New code should import directly from graph.py