#!/bin/bash
# Simple run script for the Uda-hub Autonomous AI Agent

echo "🤖 Uda-hub Autonomous AI Agent"
echo "=============================="
echo ""

# Check if .env file exists and has API key
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your OpenAI API key."
    exit 1
fi

# Check if API key is set
if grep -q "your_openai_api_key_here" .env; then
    echo "❌ Error: Please set your OpenAI API key in the .env file!"
    echo "Replace 'your_openai_api_key_here' with your actual API key."
    exit 1
fi

echo "Starting the AI agent..."
echo "Type 'quit' or 'exit' to end the conversation."
echo ""

# Run the application
python3 03_agentic_app.py