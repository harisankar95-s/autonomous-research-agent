import os
import time

from src.memory.manager import ImageStore
from src.tools.code_execution import make_execute_python_code_tool, EXECUTION_TIMEOUT_SECONDS


def test_execute_python_code(db_session, dataset_id):
    temp_files = []
    tool = make_execute_python_code_tool(temp_files, dataset_id, ImageStore(db_session))

    code = "print(2+2)"
    response = tool.func(code=code)
    assert response.strip() == "4"


def test_network_is_disabled(db_session, dataset_id):
    temp_files = []
    tool = make_execute_python_code_tool(temp_files, dataset_id, ImageStore(db_session))

    code = (
        "import urllib.request\n"
        "urllib.request.urlopen('https://www.google.com', timeout=3)\n"
        "print('network worked')"
    )
    response = tool.func(code=code)
    assert "network worked" not in response


def test_reading_app_data_csv_without_data_path_fails_fast(db_session, dataset_id):
    temp_files = []
    tool = make_execute_python_code_tool(temp_files, dataset_id, ImageStore(db_session))

    start = time.monotonic()
    response = tool.func(code="import pandas as pd\npd.read_csv('/app/data.csv')")
    elapsed = time.monotonic() - start

    assert "data_path" in response
    assert elapsed < 5  # should fail before ever touching Docker


def test_saved_plot_is_returned_as_image_and_persisted(db_session, dataset_id):
    temp_files = []
    tool = make_execute_python_code_tool(temp_files, dataset_id, ImageStore(db_session))

    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.figure(figsize=(4, 3))\n"
        "plt.plot([1, 2, 3], [1, 4, 9])\n"
        "plt.savefig('/app/output/plot.png')\n"
        "print('plot saved')"
    )
    response = tool.func(code=code)

    assert isinstance(response, dict)
    assert "plot saved" in response["text"]
    assert len(response["images"]) == 1
    image = response["images"][0]
    assert len(image["data"]) > 0
    assert image["filename"].endswith(".png")

    from src.memory.manager import AnalysisImage
    saved = db_session.query(AnalysisImage).filter_by(dataset_id=dataset_id).first()
    assert saved is not None
    assert os.path.exists(saved.file_path)
    assert saved.file_path.endswith(image["filename"])
    # no image_captions passed - falls back to a preview of the code itself
    assert saved.caption.startswith("import matplotlib")


def test_saved_plot_uses_provided_caption_instead_of_code_preview(db_session, dataset_id):
    temp_files = []
    tool = make_execute_python_code_tool(temp_files, dataset_id, ImageStore(db_session))

    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.figure(figsize=(4, 3))\n"
        "plt.plot([1, 2, 3], [1, 4, 9])\n"
        "plt.savefig('/app/output/bearing_diff.png')\n"
        "print('plot saved')"
    )
    caption = "T01 generator bearing A vs B temperature differential over time - static offset, no trend."
    response = tool.func(code=code, image_captions={"bearing_diff.png": caption})

    assert len(response["images"]) == 1

    from src.memory.manager import AnalysisImage
    saved = db_session.query(AnalysisImage).filter_by(dataset_id=dataset_id).first()
    assert saved.caption == caption
    # the file on disk gets its own unique name, independent of what the
    # sandbox script called it - captions are matched by the original name
    # before that rename, not by the final stored filename
    assert saved.file_path.endswith(response["images"][0]["filename"])


def test_long_running_code_is_terminated_by_timeout(db_session, dataset_id):
    temp_files = []
    tool = make_execute_python_code_tool(temp_files, dataset_id, ImageStore(db_session))

    code = "import time\ntime.sleep(9999)"

    start = time.monotonic()
    response = tool.func(code=code)
    elapsed = time.monotonic() - start

    assert "timeout" in response.lower()
    assert elapsed < EXECUTION_TIMEOUT_SECONDS + 30
