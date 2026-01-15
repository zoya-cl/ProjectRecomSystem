# 1. Build artifacts (already done)
python build_pipeline.py

# 2. Use CLI
python app_cli.py 

# 3. Interactive shell
python interactive_shell.py

# 4. API server
python -m uvicorn api:app --reload --port 8000
# Then visit http://localhost:8000/docs
