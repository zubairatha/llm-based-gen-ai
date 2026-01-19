#!/bin/bash
# Activation script for MCP File Analyzer project

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Virtual environment activated!"
echo "Installed packages:"
pip list --format=columns

echo ""
echo "Quick start commands:"
echo "  - Run MCP server: python main.py"
echo "  - Run demo client: python client.py"
echo "  - Interactive client: python client.py interactive"
echo ""
echo "To deactivate: deactivate"