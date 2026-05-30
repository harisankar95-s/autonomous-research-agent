from src.utils.config import config

def test_config_loads_successfully():
    assert config.gemini_api_key is not None
    assert config.gemini_model is not None
    assert config.gemini_url is not None