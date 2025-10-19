@echo off
echo ========================================
echo    🚀 PyRust-ML Enhanced Dashboard
echo ========================================
echo.
echo Starting the advanced ML toolkit...
echo.

cd /d "e:\pyrust-ml"

echo Stopping any existing dashboard instances...
taskkill /F /IM streamlit.exe >nul 2>&1

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Starting Streamlit dashboard...
echo Dashboard will open at: http://localhost:8508
echo.
echo Features available:
echo   📊 Dataset Manager - Built-in datasets + custom upload
echo   🔬 Model Playground - Interactive ML training
echo   ⚡ TRUE Rust vs Python Performance Comparison
echo   📈 Advanced Analytics - Feature importance + insights
echo.
echo Press Ctrl+C to stop the dashboard
echo ========================================
echo.

streamlit run dashboard/enhanced_app.py --server.port 8508

pause