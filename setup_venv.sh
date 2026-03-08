#!/bin/bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Virtual environment setup complete. Activate with: source venv/Scripts/activate"
