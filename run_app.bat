@echo off
REM RAG Chat Assistant Launcher Script for Windows

echo 🤖 Starting RAG-Powered Chat Assistant...
echo.

REM Check if requirements are installed
echo Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
) else (
    echo ✅ Dependencies found
)

REM Check for configuration file
if not exist "appsettings.json" (
    echo.
    echo ⚠️  Configuration file 'appsettings.json' not found!
    echo Please copy 'appsettings.sample.json' to 'appsettings.json' and configure your Azure credentials.
    echo.
    echo Commands to set up configuration:
    echo   copy appsettings.sample.json appsettings.json
    echo   # Then edit appsettings.json with your Azure credentials
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
) else (
    echo ✅ Configuration file found
)

echo.
echo 🚀 Launching Streamlit app...
echo The app will open in your default browser at http://localhost:8501
echo.

REM Launch Streamlit
streamlit run streamlit_app.py
