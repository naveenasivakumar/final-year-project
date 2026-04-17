@echo off
python -m uvicorn backend.api.routes:app --reload
pause
