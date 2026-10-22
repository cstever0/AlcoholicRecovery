"""Tests for application initialization, navigation setup, and safety disclaimers."""

import compileall
from pathlib import Path
from crip.app.components.disclaimer import render_clinical_disclaimer
from crip.app.main import navigation, pages


class TestApplicationInitialization:
    """Verify application components, pages, and navigation are properly configured."""

    def test_all_python_files_compile(self):
        """Ensure no syntax errors exist across the codebase."""
        src_path = Path("src")
        compiled = compileall.compile_dir(str(src_path), quiet=True)
        assert compiled is True, "Syntax errors detected in src directory"

    def test_navigation_pages_exist(self):
        """Ensure all required navigation pages are defined and exist on disk."""
        from crip.app.main import PAGE_CONFIGS, navigation

        expected_page_titles = {
            "Patient Profile",
            "MRI Analysis",
            "Recovery Estimate",
            "Evaluation",
        }
        actual_page_titles = {cfg["title"] for cfg in PAGE_CONFIGS}
        assert expected_page_titles == actual_page_titles

        for cfg in PAGE_CONFIGS:
            page_path = Path(cfg["file"])
            assert page_path.exists(), f"Page file does not exist: {page_path}"
            assert page_path.is_file(), f"Page path is not a file: {page_path}"

        assert navigation is not None

    def test_pages_contain_clinical_disclaimer_call(self):
        """Verify that every Streamlit page file calls render_clinical_disclaimer."""
        app_pages_dir = Path("src/crip/app/pages")
        for page_file in app_pages_dir.glob("*.py"):
            content = page_file.read_text(encoding="utf-8")
            assert "render_clinical_disclaimer" in content, (
                f"Page {page_file.name} does not import or render the clinical disclaimer"
            )

    def test_disclaimer_component_text_contains_required_clauses(self):
        """Verify the research disclaimer includes essential regulatory & safety clauses."""
        disclaimer_module_path = Path("src/crip/app/components/disclaimer.py")
        text = disclaimer_module_path.read_text(encoding="utf-8")
        assert "RESEARCH PROOF OF CONCEPT" in text
        assert "NOT FOR CLINICAL USE" in text
        assert "NOT a medical device" in text or "NOT intended for clinical diagnosis" in text

    def test_all_pages_render_with_apptest(self):
        """Verify each page renders cleanly via Streamlit AppTest without exceptions."""
        from streamlit.testing.v1 import AppTest

        root_dir = Path(__file__).resolve().parent.parent
        main_script = root_dir / "src/crip/app/main.py"

        at_main = AppTest.from_file(str(main_script), default_timeout=10)
        at_main.run()
        assert not at_main.exception

        page_files = [
            root_dir / "src/crip/app/pages/1_Patient_Profile.py",
            root_dir / "src/crip/app/pages/2_MRI_Analysis.py",
            root_dir / "src/crip/app/pages/3_Recovery_Estimate.py",
            root_dir / "src/crip/app/pages/4_Evaluation.py",
        ]
        for pf in page_files:
            at = AppTest.from_file(str(pf), default_timeout=10)
            at.run()
            assert not at.exception, f"Exception occurred while rendering {pf}: {at.exception}"
