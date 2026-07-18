import docker
import tempfile
import os

from src.tools.base import Tool
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_execute_python_code_tool(temp_files: list) -> Tool:
    def execute_python_code(code: str, data_path: str = "") -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name
        temp_files.append(temp_path)

        client = docker.from_env()

        volumes = {
            temp_path: {"bind": f"/app/{os.path.basename(temp_path)}", "mode": "ro"}
        }
        if data_path:
            volumes[data_path] = {"bind": "/app/data.csv", "mode": "ro"}

        try:
            result = client.containers.run(
                image="python-code-sandbox",
                command=f"python /app/{os.path.basename(temp_path)}",
                volumes=volumes,
                network_disabled=True,
                mem_limit="512m",
                remove=True,
                stderr=True,
            )
            output = result.decode("utf-8")
        except docker.errors.ContainerError as e:
            output = e.stderr.decode("utf-8") if e.stderr else str(e)
            logger.warning(f"Code execution failed | error={output[:200]}")

        logger.info("Code execution completed")
        return output

    return Tool(
        name="execute_python_code",
        description=(
            "Run Python code in an isolated sandbox to perform data analysis, "
            "statistical computation, or train machine learning models. "
            "pandas, numpy, scikit-learn, scipy, xgboost, and matplotlib are "
            "pre-installed. If you provide data_path (the file path returned "
            "by fetch_data), that file will be available inside your code at "
            "the fixed path /app/data.csv - read it with pandas.read_csv. "
            "Use print() statements to return any results you need to see."
        ),
        parameters={
            "code": "the Python code to execute, as a string",
            "data_path": "optional - the exact file path returned by fetch_data, if your code needs to read fetched data"
        },
        func=execute_python_code
    )