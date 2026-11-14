"""
Simple Gradio interface for DataWizard.

Run:
    python app.py
Then open http://127.0.0.1:7860 in your browser.

Tabs:
  - Chat:     ask questions. Mode "Agent" uses the tools (agent_executor),
              mode "LCEL chain" uses the plain prompt | llm | parser chain
              from the notebook, which has no tools and cannot see your data.
  - Datasets: preview any CSV in this folder, or upload your own.
"""

import glob
import os
import shutil

import gradio as gr
import pandas as pd

# Importing datawizard also sets the working directory to this folder and
# asks for the Groq API key in the terminal if it is not already set.
from datawizard import DATAFRAME_CACHE, agent_executor, analysis_chain

AGENT_MODE = "Agent (uses tools)"
CHAIN_MODE = "LCEL chain (no tools)"

EXAMPLES = [
    ["What datasets do I have?", AGENT_MODE],
    ["Which CSV files do I have, and is each one a classification or a regression dataset?", AGENT_MODE],
    ["Show me the first rows of house_prices_regression.csv", AGENT_MODE],
    ["Evaluate house_prices_regression.csv with price_usd as the target. Is it classification or regression?", AGENT_MODE],
    ["Evaluate customer_churn_classification.csv with churned as the target.", AGENT_MODE],
    ["What columns are in the dataset?", CHAIN_MODE],
]


def csv_files():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(os.getcwd(), "*.csv")))


def chat(message, history, mode):
    """Gradio chat callback. Each question is answered on its own, as in the notebook."""
    try:
        if mode == CHAIN_MODE:
            return analysis_chain.invoke({"input": message})
        result = agent_executor.invoke({"input": message})
        return result["output"]
    except Exception as e:
        return f"Error: {e}"


def preview(file_name):
    if not file_name:
        return pd.DataFrame(), "No file selected."
    df = pd.read_csv(file_name)
    return df.head(50), f"{file_name}: {df.shape[0]} rows, {df.shape[1]} columns"


def upload(file_obj):
    if file_obj is None:
        return gr.update(choices=csv_files()), "No file uploaded."
    src = file_obj if isinstance(file_obj, str) else file_obj.name
    name = os.path.basename(src)
    if not name.lower().endswith(".csv"):
        return gr.update(choices=csv_files()), "Only .csv files are supported."
    shutil.copy(src, os.path.join(os.getcwd(), name))
    DATAFRAME_CACHE.pop(name, None)  # in case a file with this name was cached
    return gr.update(choices=csv_files(), value=name), f"Uploaded {name}."


with gr.Blocks(title="DataWizard") as demo:
    gr.Markdown(
        "# DataWizard\n"
        "Ask questions about your CSV files in plain English. "
        "In Agent mode the model calls tools to list, summarize and analyze your datasets "
        "and to train and evaluate models."
    )

    with gr.Tab("Chat"):
        mode = gr.Radio(
            choices=[AGENT_MODE, CHAIN_MODE],
            value=AGENT_MODE,
            label="Mode",
        )
        gr.ChatInterface(
        fn=chat,
        additional_inputs=[mode],
        examples=EXAMPLES,
        cache_examples=False,
    )

    with gr.Tab("Datasets"):
        dataset = gr.Dropdown(choices=csv_files(), label="Dataset")
        status = gr.Markdown()
        table = gr.Dataframe(label="Preview (first 50 rows)", interactive=False)
        dataset.change(preview, dataset, [table, status])

        gr.Markdown("### Add your own CSV")
        uploader = gr.File(label="Upload a CSV", file_types=[".csv"])
        uploader.upload(upload, uploader, [dataset, status])


if __name__ == "__main__":
    demo.launch()
