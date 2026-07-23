import os
import time
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.memory.manager import ImageStore
from src.tools.code_execution import make_execute_python_code_tool, EXECUTION_TIMEOUT_SECONDS
from src.utils.config import config


def _tool(temp_files, dataset_id=None):
    dataset_id = dataset_id or str(uuid.uuid4())
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()
    image_store = ImageStore(db_session)
    return make_execute_python_code_tool(temp_files, dataset_id, image_store), dataset_id, db_session


def test_execute_python_code():
    temp_files = []
    tool, _, db_session = _tool(temp_files)

    code = "print(2+2)"
    response = tool.func(code=code)
    assert response.strip() == "4"

    db_session.close()


def test_network_is_disabled():
    temp_files = []
    tool, _, db_session = _tool(temp_files)

    code = (
        "import urllib.request\n"
        "urllib.request.urlopen('https://www.google.com', timeout=3)\n"
        "print('network worked')"
    )
    response = tool.func(code=code)
    assert "network worked" not in response

    db_session.close()


def test_reading_app_data_csv_without_data_path_fails_fast():
    temp_files = []
    tool, _, db_session = _tool(temp_files)

    start = time.monotonic()
    response = tool.func(code="import pandas as pd\npd.read_csv('/app/data.csv')")
    elapsed = time.monotonic() - start

    assert "data_path" in response
    assert elapsed < 5  # should fail before ever touching Docker

    db_session.close()


def test_saved_plot_is_returned_as_image_and_persisted():
    temp_files = []
    dataset_id = str(uuid.uuid4())
    tool, dataset_id, db_session = _tool(temp_files, dataset_id)

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

    db_session.close()


def test_long_running_code_is_terminated_by_timeout():
    temp_files = []
    tool, _, db_session = _tool(temp_files)

    code = "import time\ntime.sleep(9999)"

    start = time.monotonic()
    response = tool.func(code=code)
    elapsed = time.monotonic() - start

    assert "timeout" in response.lower()
    assert elapsed < EXECUTION_TIMEOUT_SECONDS + 30

    db_session.close()
