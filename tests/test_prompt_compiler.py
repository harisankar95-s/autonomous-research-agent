from src.prompt_building.compiler import compile_prompt


def test_compile_prompt_minimal():
    result = compile_prompt(
        role_prompt="You are a data scientist.",
        project_brief="Analyze solar panel data."
    )

    assert "You are a data scientist." in result
    assert "Analyze solar panel data." in result
    assert "KNOWN FACTS" not in result
    assert "RELEVANT PRIOR FINDINGS" not in result


def test_compile_prompt_full():
    result = compile_prompt(
        role_prompt="You are a data scientist.",
        project_brief="Analyze solar panel data.",
        facts="Task type: regression",
        knowledge=["Column A is skewed", "Column B has missing values"]
    )

    assert "You are a data scientist." in result
    assert "Analyze solar panel data." in result
    assert "KNOWN FACTS" in result
    assert "Task type: regression" in result
    assert "RELEVANT PRIOR FINDINGS" in result
    assert "Column A is skewed" in result
    assert "Column B has missing values" in result


def test_compile_prompt_always_embeds_always_load_skill_bodies():
    result = compile_prompt(
        role_prompt="You are a data scientist.",
        project_brief="Analyze solar panel data."
    )

    assert "METHODOLOGY STANDARDS" in result
    assert "GROUNDED THRESHOLDS" in result
    assert "OUTLIERS VS ANOMALIES" in result
    # always-load skills shouldn't also be advertised in the discoverable catalog
    assert "general_statistical_rigor:" not in result