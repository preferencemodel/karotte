from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent.parent / "src/karotte/templates"


def test_base_template_exposes_the_uv_extra_block() -> None:
    """Venvs extend `[tool.uv]` through this block; a second table is a duplicate-key error in uv."""
    base = (TEMPLATES / "pyproject.base.toml.jinja").read_text()
    assert "{% block uv_extra %}{% endblock %}" in base
    assert base.index("[tool.uv]") < base.index("{% block uv_extra %}")
    assert base.index("{% block uv_extra %}") < base.index("[[tool.uv.index]]")
