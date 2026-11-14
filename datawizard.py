"""
DataWizard: AI-Powered Data Analysis (Groq edition)

This file is the notebook "LLM-Powered_Data_Science-RomanUrdu-Groq.ipynb"
converted into a single Python file. The tools, prompt, agent, executor and
LCEL chain are the same code as in the notebook, in the same order.

Run in the terminal:
    python datawizard.py          # chat loop (type 'exit' to quit)
    python datawizard.py --demo   # runs the notebook's demo cells once

The Gradio interface (app.py) imports `agent_executor` and `analysis_chain`
from this file.

Groq API key: set GROQ_API_KEY in your environment or in a .env file next to
this file. If it is missing, you will be asked for it in the terminal.
"""

import glob
import os
import sys
import getpass
from typing import List, Optional, Dict, Any

import pandas as pd

# list_csv_files() looks in the current working directory, so make sure that is
# the folder this file lives in (where the CSV files are), wherever you start it.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Groq API key
# --------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# The key is asked only once (if it is not already set in the environment)
if "GROQ_API_KEY" not in os.environ:
    if sys.stdin is not None and sys.stdin.isatty():
        os.environ["GROQ_API_KEY"] = getpass.getpass("Paste your Groq API key: ")
    else:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a free key at https://console.groq.com "
            "and put it in the environment or in a .env file."
        )

# --------------------------------------------------------------------------
# LangChain tools
# --------------------------------------------------------------------------
from langchain_core.tools import tool


@tool
def list_csv_files() -> Optional[List[str]]:
    """List all CSV file names in the local directory.

    Returns:
        A list containing CSV file names.
        If no CSV files are found, returns None.
    """
    csv_files = glob.glob(os.path.join(os.getcwd(), "*.csv"))
    if not csv_files:
        return None
    return [os.path.basename(file) for file in csv_files]


DATAFRAME_CACHE = {}


@tool
def preload_datasets(paths: List[str]) -> str:
    """
    Loads CSV files into a global cache if not already loaded.

    This function helps to efficiently manage datasets by loading them once
    and storing them in memory for future use. Without caching, you would
    waste tokens describing dataset contents repeatedly in agent responses.

    Args:
        paths: A list of file paths to CSV files.

    Returns:
        A message summarizing which datasets were loaded or already cached.
    """
    loaded = []
    cached = []
    for path in paths:
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)
            loaded.append(path)
        else:
            cached.append(path)

    return (
        f"Loaded datasets: {loaded}\n"
        f"Already cached: {cached}"
    )


@tool
def get_dataset_summaries(dataset_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze multiple CSV files and return metadata summaries for each.

    Args:
        dataset_paths (List[str]):
            A list of file paths to CSV datasets.

    Returns:
        List[Dict[str, Any]]:
            A list of summaries, one per dataset, each containing:
            - "file_name": The path of the dataset file.
            - "column_names": A list of column names in the dataset.
            - "data_types": A dictionary mapping column names to their data types (as strings).
    """
    summaries = []

    for path in dataset_paths:
        # Load and cache the dataset if it is not cached yet
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)

        df = DATAFRAME_CACHE[path]

        summary = {
            "file_name": path,
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
        }
        summaries.append(summary)

    return summaries


@tool
def call_dataframe_method(file_name: str, method: str) -> str:
    """
    Execute a method on a DataFrame and return the result.
    This tool lets you run simple DataFrame methods like 'head', 'tail', or 'describe'
    on a dataset that has already been loaded and cached using 'preload_datasets'.
    Args:
        file_name (str): The path or name of the dataset in the global cache.
        method (str): The name of the method to call on the DataFrame. Only no-argument
                      methods are supported (e.g., 'head', 'describe', 'info').
    Returns:
        str: The output of the method as a formatted string, or an error message if
             the dataset is not found or the method is invalid.
    Example:
        call_dataframe_method(file_name="data.csv", method="head")
    """
    # Take it from the cache, otherwise try to load it from disk
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return f"DataFrame '{file_name}' not found in cache or on disk."
        except Exception as e:
            return f"Error loading '{file_name}': {str(e)}"

    df = DATAFRAME_CACHE[file_name]
    func = getattr(df, method, None)
    if not callable(func):
        return f"'{method}' is not a valid method of DataFrame."
    try:
        result = func()
        return str(result)
    except Exception as e:
        return f"Error calling '{method}' on '{file_name}': {str(e)}"


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


def _get_df(file_name: str):
    """Helper: return the DataFrame from cache or disk. Returns (df, error)."""
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return None, f"DataFrame '{file_name}' not found in cache or on disk."
        except Exception as e:
            return None, f"Error loading '{file_name}': {str(e)}"
    return DATAFRAME_CACHE[file_name], None


@tool
def evaluate_classification_dataset(file_name: str, target_column: str) -> Dict[str, Any]:
    """
    Train and evaluate a classifier on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the classification target.
    Returns:
        Dict[str, Any]: A dictionary with the model's accuracy score, or an error message.
    """
    df, err = _get_df(file_name)
    if err:
        return {"error": err}
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}

    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {"accuracy": float(accuracy_score(y_test, y_pred))}
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}


@tool
def evaluate_regression_dataset(file_name: str, target_column: str) -> Dict[str, Any]:
    """
    Train and evaluate a regression model on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the regression target.
    Returns:
        Dict[str, Any]: A dictionary with R2 score and Mean Squared Error, or an error message.
    """
    df, err = _get_df(file_name)
    if err:
        return {"error": err}
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}

    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "r2_score": float(r2_score(y_test, y_pred)),
            "mean_squared_error": float(mean_squared_error(y_test, y_pred)),
        }
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data science assistant. Use the available tools to analyze CSV files. "
     "Your job is to determine whether each dataset is for classification or regression, based on its structure."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # required for tool-calling agents
])

from langchain_groq import ChatGroq

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

tools = [
    list_csv_files,
    preload_datasets,
    get_dataset_summaries,
    call_dataframe_method,
    evaluate_classification_dataset,
    evaluate_regression_dataset,
]

from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
)
agent_executor.agent.stream_runnable = False

# --------------------------------------------------------------------------
# LCEL chain
# --------------------------------------------------------------------------
from langchain_core.output_parsers import StrOutputParser

lcel_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data science assistant. Answer questions clearly and concisely."),
    ("user", "{input}"),
])

# Chain: prompt -> language model -> string parser
analysis_chain = lcel_prompt | llm | StrOutputParser()

# Reuse the chain under another name if needed
chain = analysis_chain


# --------------------------------------------------------------------------
# Notebook demo cells (only run with --demo)
# --------------------------------------------------------------------------
def run_demo():
    csvs = list_csv_files.invoke({}) or []

    print("Tool Name:", list_csv_files.name)
    print("Tool Description:", list_csv_files.description)
    print("Tool Arguments:", list_csv_files.args)

    print(list_csv_files.invoke({}))
    if csvs:
        print(preload_datasets.invoke({"paths": csvs}))
        print(get_dataset_summaries.invoke({"dataset_paths": [csvs[0]]}))

    print(llm.invoke("Say hello in one short sentence.").content)

    # Raw agent: only one step, no tool loop
    agent_response = agent.invoke({
        "input": "What columns are in the dataset?",
        "intermediate_steps": [],
    })
    print("Raw agent response:")
    print(agent_response)

    if isinstance(agent_response, list) and agent_response:
        action = agent_response[0]
        if hasattr(action, "tool"):
            print("The agent decided to call a tool:")
            print("Tool Name:", action.tool)
            print("Tool Input:", action.tool_input)
        else:
            print("Agent returned:", action)
    elif hasattr(agent_response, "return_values"):
        print("The agent gave a final response:")
        print(agent_response.return_values)
    else:
        print("Unexpected agent response type:", type(agent_response).__name__)
        print(agent_response)

    # Executor: full ReAct loop
    result = agent_executor.invoke({
        "input": "Which CSV files do I have, and is each one a classification or a regression dataset?"
    })
    print("\nFinal answer:\n", result["output"])

    # LCEL chain
    print("LCEL chain running:")
    lcel_response = analysis_chain.invoke({
        "input": "What columns are in the dataset?"
    })
    print(lcel_response[:400])


# --------------------------------------------------------------------------
# Terminal chat loop (same as the notebook)
# --------------------------------------------------------------------------
def main():
    print("Ask questions about your datasets ('exit' to quit):")

    while True:
        user_input = input(" You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("See you later.")
            break

        try:
            result = agent_executor.invoke({"input": user_input})
            print(f"DataWizard: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        main()
