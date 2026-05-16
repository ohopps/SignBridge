@echo off
echo Starting HealthComm AI Assistant...
python -m streamlit run app.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo ========================================
    echo ERROR: Failed to start Streamlit!
    echo Please make sure you have installed streamlit:
    echo run 'pip install streamlit' in your terminal.
    echo ========================================
    pause
)
