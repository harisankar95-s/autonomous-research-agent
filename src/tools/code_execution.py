import base64
import docker
import glob
import shutil
import tempfile
import os

import requests

from src.memory.manager import ImageStore, _as_dict
from src.tools.base import Tool
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXECUTION_TIMEOUT_SECONDS = 60
MAX_IMAGES_RETURNED = 6
CAPTION_PREVIEW_LENGTH = 200


def make_execute_python_code_tool(temp_files: list, dataset_id: str, image_store: ImageStore) -> Tool:
    def execute_python_code(code: str, data_path: str = "", image_captions: dict | None = None) -> str | dict:
        image_captions = _as_dict(image_captions)
        if data_path and not os.path.exists(data_path):
            error_msg = (
                f"Error: data_path '{data_path}' does not exist on the host "
                f"filesystem. data_path must be the exact file path returned "
                f"by fetch_data - a real file on this machine - not a path "
                f"inside the sandbox such as /app/data.csv."
            )
            logger.warning(f"Invalid data_path | path={data_path}")
            return error_msg

        if "/app/data.csv" in code and not data_path:
            error_msg = (
                "Error: your code reads /app/data.csv, but you did not pass "
                "data_path in this same tool call. Without data_path, no data "
                "file exists in the sandbox at all - /app/data.csv will not be "
                "there. Call execute_python_code again with the data_path "
                "argument set to the exact file path fetch_data returned."
            )
            logger.warning("Code references /app/data.csv without data_path")
            return error_msg

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name
        temp_files.append(temp_path)

        output_dir = tempfile.mkdtemp()

        client = docker.from_env()

        volumes = {
            temp_path: {"bind": f"/app/{os.path.basename(temp_path)}", "mode": "ro"},
            output_dir: {"bind": "/app/output", "mode": "rw"},
        }
        if data_path:
            volumes[data_path] = {"bind": "/app/data.csv", "mode": "ro"}

        output = ""
        container = None
        try:
            container = client.containers.run(
                image="python-code-sandbox",
                command=f"python /app/{os.path.basename(temp_path)}",
                volumes=volumes,
                network_disabled=True,
                mem_limit="512m",
                detach=True,
            )
            try:
                result = container.wait(timeout=EXECUTION_TIMEOUT_SECONDS)
                output = container.logs(stdout=True, stderr=True).decode("utf-8")
                if result.get("StatusCode", 0) != 0:
                    logger.warning(f"Code execution failed | error={output[:200]}")
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                output = (
                    f"Error: code execution exceeded the "
                    f"{EXECUTION_TIMEOUT_SECONDS}s timeout and was terminated. "
                    f"Simplify the code or process less data per call."
                )
                logger.warning("Code execution timed out")
        except docker.errors.APIError as e:
            output = f"Docker error: {str(e)}"
            logger.warning(f"Docker API error | error={output[:200]}")
        finally:
            if container is not None:
                try:
                    container.kill()
                except Exception:
                    pass
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        fallback_caption = code[:CAPTION_PREVIEW_LENGTH]
        images = []
        for img_path in sorted(glob.glob(os.path.join(output_dir, "*.png")))[:MAX_IMAGES_RETURNED]:
            filename_on_disk = os.path.basename(img_path)
            caption = image_captions.get(filename_on_disk) or fallback_caption
            with open(img_path, "rb") as img_f:
                image_b64 = base64.b64encode(img_f.read()).decode("ascii")
            record = image_store.save_image(dataset_id, image_b64, caption=caption)
            images.append({"data": image_b64, "filename": os.path.basename(record.file_path)})
        shutil.rmtree(output_dir, ignore_errors=True)

        logger.info(f"Code execution completed | images={len(images)}")

        if images:
            return {"text": output, "images": images}
        return output

    return Tool(
        name="execute_python_code",
        description=(
            "Run Python code in an isolated sandbox to perform data analysis, "
            "statistical computation, or train machine learning models. "
            "pandas, numpy, scikit-learn, scipy, xgboost, and matplotlib are "
            "pre-installed. To read data fetched by fetch_data, you MUST pass "
            "data_path in this same tool call, set to the exact file path "
            "fetch_data returned - only then will that file exist inside your "
            "code at /app/data.csv, readable with pandas.read_csv. If you "
            "omit data_path, /app/data.csv does not exist at all, whether or "
            "not you fetched data earlier - there is no persistent state "
            "between calls. "
            "Use print() statements to return any numeric or text results you "
            "need to see. To visually inspect a distribution, relationship, or "
            "pattern, save a figure with plt.savefig('/app/output/<name>.png') "
            "- any PNGs saved there are returned to you as images you can "
            "actually view, so use this for real visual inspection rather than "
            "only printed summary statistics. Keep figures small (e.g. "
            "figsize=(6,4), dpi=80) and save at most a few per call. Whenever "
            "you save a figure, also pass image_captions describing it - this "
            "caption is what every future run will see when deciding whether "
            "this image is worth looking at again, so make it specific (what "
            "it shows, which entity/column, what it's evidence for), not a "
            "generic label. Code that runs longer than 60 seconds will be "
            "terminated - keep computations bounded."
        ),
        parameters={
            "code": "the Python code to execute, as a string",
            "data_path": "optional - the exact file path returned by fetch_data, if your code needs to read fetched data",
            "image_captions": "optional - a mapping from each exact filename you saved (e.g. 'bearing_temp.png') to a one to two sentence description of what it shows and why you made it"
        },
        func=execute_python_code
    )