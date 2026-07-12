"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documenation builds.
"""
# ruff: noqa: F403, F405

from pathlib import Path

from documenteer.conf.guide import *

from lsst.images.schema_docs import generate_schema_docs

exclude_patterns.append("changes/*")

autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_model_show_json = False

# Generate one page per frozen JSON schema so that the canonical schema URLs
# (https://images.lsst.io/schemas/{name}-{version}) resolve, and stage the raw
# JSON files for publication alongside the pages.
_doc_dir = Path(__file__).parent
generate_schema_docs(
    schema_dir=_doc_dir.parent / "schemas",
    page_dir=_doc_dir / "schemas",
    extra_dir=_doc_dir / "_extra",
)
html_extra_path = ["_extra"]
