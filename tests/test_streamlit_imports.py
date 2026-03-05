def test_app_imports_without_streamlit():
    # Importing the entrypoint module should not require streamlit to be installed.
    import app.main  # noqa: F401

