<div align="center">

# 🏭 Manufacturing Analytics Copilot

**Ask questions about manufacturing data in plain English — no SQL required**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://mfg-analytics-copilot.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-85%2B_Passing-success?style=for-the-badge)](tests/)

<br/>

<img src="https://img.shields.io/badge/LLM-Groq_API-orange?style=flat-square" alt="Groq">
<img src="https://img.shields.io/badge/Database-DuckDB-FFF000?style=flat-square" alt="DuckDB">
<img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square" alt="Streamlit">
<img src="https://img.shields.io/badge/Charts-Plotly-3F4F75?style=flat-square" alt="Plotly">

</div>

---

## 🎯 What is this?

A **natural language interface** for querying manufacturing sensor data. Plant engineers can ask questions like:

> *"What is the overall machine failure rate?"*  
> *"Which machine type fails most often?"*  
> *"How many machines need immediate maintenance?"*

The system converts questions to SQL, validates for safety, executes against DuckDB, and returns results with **auto-generated visualizations**.

<br/>

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🗣️ **Natural Language Queries** | Ask questions in plain English — the LLM handles SQL generation |
| 🛡️ **SQL Safety Validation** | Blocks DML operations, injection attempts, and unknown tables |
| 📊 **Auto Chart Selection** | Automatically picks the best visualization (metric card, bar, scatter, table) |
| 💬 **Smart Greeting Handler** | Responds to "hi" and "help" instantly without LLM calls |
| 📖 **Semantic Layer** | Business glossary + schema definitions + few-shot examples for accurate SQL |
| 🔌 **Multi-Provider LLM** | Supports Groq, AWS Bedrock, and Ollama via factory pattern |

<br/>

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com/keys) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/antimoh/manufacturing-analytics-copilot.git
cd manufacturing-analytics-copilot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

<br/>

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Question                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Greeting Handler    →  Instant response (0 tokens)        │
├─────────────────────────────────────────────────────────────┤
│  Semantic Layer      →  Schema + Glossary + Examples       │
├─────────────────────────────────────────────────────────────┤
│  Prompt Engine       →  Constructs SQL generation prompt   │
├─────────────────────────────────────────────────────────────┤
│  LLM Client          →  Groq / Bedrock / Ollama            │
├─────────────────────────────────────────────────────────────┤
│  SQL Validator       →  Blocks DML & injection attacks     │
├─────────────────────────────────────────────────────────────┤
│  Query Executor      →  DuckDB execution                   │
├─────────────────────────────────────────────────────────────┤
│  Chart Selector      →  Auto-selects visualization         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Response: Explanation + Chart + SQL              │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Data engineering first** — LLM sits on a well-modeled semantic layer, not raw tables
2. **Transparency by default** — Always show generated SQL alongside the answer
3. **Fail safely** — Gracefully reject bad SQL from the LLM
4. **Scope enforcement** — Only answer questions about available data
5. **No LangChain** — Custom orchestration for full transparency and testability

<br/>

## 📊 Data Source

Built on the [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset):

| Metric | Value |
|--------|-------|
| Sensor readings | 10,000 |
| Machine failures | 339 (3.39%) |
| Machine types | 3 (H, M, L) |
| Failure modes | 5 types |

### Failure Types

| Code | Name | Description |
|------|------|-------------|
| **HDF** | Heat Dissipation Failure | Temperature differential issues |
| **OSF** | Overstrain Failure | Torque × tool wear exceeds threshold |
| **PWF** | Power Failure | Power outside 3500W–9000W range |
| **TWF** | Tool Wear Failure | Tool wear > 200-240 min |
| **RNF** | Random Failure | 0.1% baseline probability |

<br/>

## 🛡️ SQL Safety

The validator protects against:

- ✅ DML operations (INSERT, UPDATE, DELETE, DROP)
- ✅ SQL injection (comments, UNION, stacked queries)
- ✅ Unknown table references
- ✅ Timing attacks (SLEEP, BENCHMARK)
- ✅ File system access (INTO OUTFILE, LOAD_FILE)

**Only SELECT queries against 5 allowed tables are permitted.**

<br/>

## 🧪 Testing

```bash
# Run all 85+ tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=copilot --cov-report=html

# Run golden dataset evaluation (37 test cases)
python tests/test_golden_evaluation.py
```

### Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_sql_validator.py` | 43 | DML blocking, injection prevention |
| `test_query_executor.py` | 22 | Known values, edge cases |
| `test_chart_selector.py` | 20 | Type inference, truncation |

<br/>

## ⚙️ Configuration

```bash
# Required
GROQ_API_KEY=gsk_...              # Get from console.groq.com/keys

# Optional
LLM_PROVIDER=groq                 # groq | bedrock | ollama
GROQ_MODEL=openai/gpt-oss-20b    # Override default model
```

<br/>

## 📁 Project Structure

```
manufacturing-analytics-copilot/
├── copilot/                 # Core library
│   ├── semantic_layer.py    # Schema + glossary + examples
│   ├── prompt_engine.py     # Prompt construction
│   ├── llm_client.py        # LLM provider abstraction
│   ├── sql_validator.py     # Safety validation
│   ├── query_executor.py    # DuckDB wrapper
│   ├── orchestrator.py      # Main pipeline
│   └── chart_selector.py    # Auto chart selection
├── data/                    # CSV data files
├── tests/                   # Test suite
├── docs/                    # Documentation
├── app.py                   # Streamlit app
└── requirements.txt
```

<br/>

## 🔗 Related Projects

This is **Project 2** in my data engineering portfolio:

| Project | Description | Stack |
|---------|-------------|-------|
| [Manufacturing Analytics Platform](https://github.com/antimoh/manufacturing-analytics-platform) | Data engineering foundation | Python, Snowflake, dbt |
| **Manufacturing Analytics Copilot** | Natural language interface | LLM, DuckDB, Streamlit |

<br/>

## 📚 References

- [dbt: Semantic Layer vs Text-to-SQL](https://docs.getdbt.com/blog/semantic-layer-vs-text-to-sql-2026)
- [Best Practices for LLM-Generated SQL](https://medium.com/@vi.ha.engr/bridging-natural-language-and-databases-best-practices-for-llm-generated-sql-fcba0449d4e5)
- [Why Your AI Agent Needs a Semantic Layer](https://xomnia.com/post/text-to-sql-with-llms-why-your-ai-agent-needs-a-semantic-layer/)

<br/>

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

<br/>

## 👤 Author

**Anton Timokhin**  
Senior Data Analyst at Align Technology  
AWS Certified Data Engineer — Associate

[![GitHub](https://img.shields.io/badge/GitHub-antimoh-181717?style=flat-square&logo=github)](https://github.com/antimoh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-antimoh-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/antimoh)

---

<div align="center">

*"Making data accessible to everyone while maintaining the transparency that data professionals require."*

</div>