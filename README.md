# Manufacturing Analytics Copilot 🏭

A natural language interface that lets plant engineers query manufacturing sensor data and predictive maintenance insights using plain English, without writing SQL.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What It Does

Ask questions like:
- "What is the overall machine failure rate?"
- "Which machine type fails most often?"
- "How many machines need immediate maintenance?"

The copilot:
1. **Understands greetings** — responds helpfully to "hi"
2. **Converts questions to SQL** using an LLM with semantic context
3. **Validates SQL for safety** — blocks all DML and injection attempts
4. **Executes against DuckDB** — fast, in-process analytics
5. **Explains results** in plain English
6. **Auto-generates charts** — metric cards, bar charts, scatter plots, or tables

## 📊 Data Source

Built on the UCI AI4I 2020 Predictive Maintenance Dataset:
- 10,000 sensor readings
- 339 machine failures (3.39%)
- 3 machine types (H, M, L)
- 5 failure modes (TWF, HDF, PWF, OSF, RNF)

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| LLM Inference | Groq API (GPT-OSS 20B) |
| Local Data Query | DuckDB |
| Frontend | Streamlit |
| Visualization | Plotly |
| Testing | pytest (85+ tests) |

## 🚀 Quick Start

```bash
git clone https://github.com/antimoh/manufacturing-analytics-copilot.git
cd manufacturing-analytics-copilot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
streamlit run app.py
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📁 Project Structure

```
manufacturing-analytics-copilot/
├── copilot/                    # Core library
│   ├── semantic_layer.py       # Schema + glossary + examples
│   ├── prompt_engine.py        # Prompt construction
│   ├── llm_client.py           # Provider abstraction
│   ├── sql_validator.py        # Safety layer
│   ├── query_executor.py       # DuckDB wrapper
│   ├── orchestrator.py         # Main pipeline
│   └── chart_selector.py       # Auto chart selection
├── data/                       # Mart tables (CSV)
├── tests/                      # Test suite + golden dataset
├── docs/                       # Architecture documentation
├── app.py                      # Streamlit application
└── requirements.txt
```

## 📝 License

MIT License

## 👤 Author

**Anton Timokhin** - Senior Data Analyst at Align Technology
