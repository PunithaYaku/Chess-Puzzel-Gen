#!/bin/bash
# One-click environment setup script

echo "Setting up AI Chess Puzzle Generator environment..."

if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Virtual environment created."
fi

source venv/bin/activate || source venv/Scripts/activate

pip install -r requirements.txt

echo "Installation complete. Downloading sample data..."
python download_data.py

echo "Setup finished. You can now run the app with: python app.py"
