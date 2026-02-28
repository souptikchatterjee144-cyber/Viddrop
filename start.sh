#!/bin/bash

echo "Checking for Python 3..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "  Python 3 not found!"
    echo "  Install Python 3.8+ from https://python.org"
    echo ""
    exit 1
fi

echo "Installing dependencies..."
pip3 install -r requirements.txt -q

clear
echo "================================"
echo "  VidDrop Companion Starting..."
echo "================================"
echo ""

python3 companion.py
