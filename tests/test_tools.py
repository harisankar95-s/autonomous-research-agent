from src.tools.search import web_search_tool

def test_web_search_returns_results():
    results = web_search_tool.func("NVIDIA earnings 2025")
    assert results is not None
    assert len(results) > 0

def test_web_search_tool_schema():
    assert web_search_tool.name == "web_search"
    assert web_search_tool.description is not None
    assert web_search_tool.func is not None