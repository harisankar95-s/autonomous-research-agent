from src.tools.code_execution import make_execute_python_code_tool


def test_execute_python_code():
    temp_files = []
    tool = make_execute_python_code_tool(temp_files)

    code = "print(2+2)"
    response = tool.func(code=code)
    assert response.strip() == "4"


def test_network_is_disabled():
    temp_files = []
    tool = make_execute_python_code_tool(temp_files)

    code = (
        "import urllib.request\n"
        "urllib.request.urlopen('https://www.google.com', timeout=3)\n"
        "print('network worked')"
    )
    response = tool.func(code=code)
    assert "network worked" not in response