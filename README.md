# DataWizard (notebook ported to Python)

- `datawizard.py` : the notebook code in one file (tools, agent, executor, LCEL chain)
- `app.py`        : Gradio interface
- `*.csv`         : 7 sample datasets (4 regression, 3 classification, all numeric)
- `requirements.txt`, `.env.example`

## Setup

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

Create a free key at https://console.groq.com, copy `.env.example` to `.env`
and paste the key. (If you skip this, the program asks for the key in the terminal.)

## Run

    python app.py             # web interface at http://127.0.0.1:7860
    python datawizard.py      # terminal chat loop
    python datawizard.py --demo   # the notebook's demo cells, once

## Datasets

| File | Task | Target column |
|---|---|---|
| house_prices_regression.csv | regression | price_usd |
| student_performance_regression.csv | regression | final_score |
| weekly_sales_regression.csv | regression | weekly_sales |
| employee_salary_regression.csv | regression | salary_usd |
| customer_churn_classification.csv | classification | churned |
| loan_approval_classification.csv | classification | approved |
| flower_species_classification.csv | classification | species |

The training tools are unchanged from the notebook, so they need numeric
feature columns. Categories in the sample files are already stored as codes.
