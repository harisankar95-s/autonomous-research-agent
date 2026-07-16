import docker
import tempfile
import os

from src.tools.base import Tool
from src.utils.logger import get_logger

logger = get_logger(__name__)


def execute_python_code(code: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    client = docker.from_env()

    try:
        result = client.containers.run(
            image="python-code-sandbox",
            command=f"python /app/{os.path.basename(temp_path)}",
            volumes={temp_path: {"bind": f"/app/{os.path.basename(temp_path)}", "mode": "ro"}},
            network_disabled=True,
            mem_limit="512m",
            remove=True,
            stderr=True,
        )
        output = result.decode("utf-8")
    except docker.errors.ContainerError as e:
        output = e.stderr.decode("utf-8") if e.stderr else str(e)
        logger.warning(f"Code execution failed | error={output[:200]}")

    os.remove(temp_path)
    logger.info("Code execution completed")
    return output


execute_python_code_tool = Tool(
    name="execute_python_code",
    description=(
        "Run Python code in an isolated sandbox to perform data analysis, "
        "statistical computation, or train machine learning models. "
        "pandas, numpy, scikit-learn, scipy, xgboost, and matplotlib are "
        "pre-installed. Use this instead of reasoning about calculations in "
        "text whenever exact numeric results are needed. The sandbox has no "
        "network or file access beyond the code itself, so use print() "
        "statements to return any results you need to see."
    ),
    parameters={"code": "the Python code to execute, as a string"},
    func=execute_python_code
)