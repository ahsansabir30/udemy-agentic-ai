# Document Assistant Project

The Document Assistant project is utilising an AI-powered document processing system using LangChain and LangGraph.

It uses a multi-agent architecture to handle different tasks, including answering questions about documents, generating summaries and key points, and performing calculations on financial and healthcare data.

## Getting Started

### Dependencies

Make sure you have the following installed:

- Python 3.9+
- Git
- pip

### Installation

```bash
git clone https://github.com/username/repository-name.git
cd report-building-agent

python -m venv venv

source venv/bin/activate

# venv\Scripts\Activate.ps1 (Windows)

pip install -r requirements.txt

cp .env.example .env # update parameters in this file 

python main.py
```

1. Clone the repository:
```bash

git clone https://github.com/username/repository-name.git
cd report-building-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows -> venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
# Edit file with env variables
cp .env.example .env
```

### Running the Assistant

```bash
python main.py
```

