#!/bin/bash

# RAG Chat Assistant Launcher Script

echo "🤖 Starting RAG-Powered Chat Assistant..."
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
else
    echo "✅ Dependencies found"
fi

# Check for configuration file
if [ ! -f "appsettings.json" ]; then
    echo ""
    echo "⚠️  Configuration file 'appsettings.json' not found!"
    echo "Please copy 'appsettings.sample.json' to 'appsettings.json' and configure your Azure credentials."
    echo ""
    echo "Commands to set up configuration:"
    echo "  cp appsettings.sample.json appsettings.json"
    echo "  # Then edit appsettings.json with your Azure credentials"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Configuration file found"
fi

echo ""
echo "🚀 Launching Streamlit app..."
echo "The app will open in your default browser at http://localhost:8501"
echo ""

# Launch Streamlit
streamlit run streamlit_app.py
