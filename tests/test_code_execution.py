from src.tools.code_execution import execute_python_code


def test_execute_python_code():
    code = "print(2+2)"
    response = execute_python_code(code)
    assert response.strip() == "4"

def test_network_is_disabled():
    code = (
        "import urllib.request\n"
        "urllib.request.urlopen('https://www.google.com', timeout=3)\n"
        "print('network worked')"
    )
    response = execute_python_code(code)
    assert "network worked" not in response