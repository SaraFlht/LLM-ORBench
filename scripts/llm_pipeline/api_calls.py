"""
api_calls.py
-----------
LLM-based question answering and reasoning over ontologies using OpenAI, DeepSeek, and OpenRouter models.
Supports both pandas DataFrame and CSV workflows.
Import and use the main function: run_llm_reasoning(df, ontology_base_path, ...)
"""
import os
import time
import sys
from datetime import datetime
import openai
from dotenv import load_dotenv
import concurrent.futures
from threading import Lock
import pandas as pd

load_dotenv()

# Model dictionary (display name → API model ID)
MODELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "deepseek-reasoner": "deepseek-reasoner",
    "llama-4-maverick": "meta-llama/llama-4-maverick"
}

def get_model_params():
    return {
        "gpt-4o-mini": {
            "temperature": 0.0,
            "max_tokens": 4096,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        },
        "deepseek-reasoner": {
            "temperature": 0.0,
            "max_tokens": 4096,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        },
        "llama-4-maverick": {
            "temperature": 0.0,
            "max_tokens": 4096,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }
    }

# Initialize clients
openai_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# OpenAI client
openai_client = openai.OpenAI(api_key=openai_api_key)

# DeepSeek client
deepseek_client = openai.OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)

# OpenRouter client
openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key
)

def get_model_version_info(response, model_id):
    model_version = "Unknown"
    if hasattr(response, 'model'):
        model_version = response.model
    if hasattr(response, 'system_fingerprint'):
        model_version += f" (fingerprint: {response.system_fingerprint})"
    return model_version

def clean_input_dataframe(df):
    """Drop unnecessary columns from input DataFrame"""
    columns_to_drop = [
        "Task ID temp",
        "Bin_Size of ontology ABox",
        "Bin_Avg Min Explanation Size",
        "strata",
        "split"
    ]

    # Drop columns that exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)
        print(f"Dropped columns: {existing_columns_to_drop}")

    return df

def run_llm_reasoning(df, ontology_base_path, models=MODELS, model_params=None, log_each=False, context_mode="ttl", show_qa=True, max_workers=3, batch_size=10, question_column="Question"):
    """
    Optimized version with parallel processing using multiple API providers
    For each question in the DataFrame, run LLM reasoning using the specified models.
    Adds response columns for each model.
    Args:
        df: pandas DataFrame with columns ['Question', 'Root Entity', 'Answer Type'] or ['SPARQL Query', 'Root Entity', 'Answer Type']
        ontology_base_path: directory containing ontology .ttl or .json files
        models: dict of display_name → model_id
        model_params: dict of display_name → generation params
        log_each: if True, print progress for each query
        context_mode: 'ttl' (default) or 'json' (for verbalized ontologies)
        show_qa: if True, display questions and answers in terminal
        max_workers: number of parallel workers for processing
        batch_size: batch size for processing (currently for future use)
        question_column: name of column containing questions ("Question" or "SPARQL Query")
    Returns:
        df: DataFrame with new response columns
        logs: list of dicts with detailed logs
    """
    # Clean input DataFrame
    df = clean_input_dataframe(df)

    # Validate question column exists
    if question_column not in df.columns:
        available_cols = [col for col in df.columns if col in ["Question", "SPARQL Query"]]
        if available_cols:
            question_column = available_cols[0]
            print(f"Using '{question_column}' column for questions")
        else:
            raise ValueError(f"DataFrame must contain '{question_column}' column or one of: ['Question', 'SPARQL Query']")

    if model_params is None:
        model_params = get_model_params()

    # Initialize response columns
    for display_name in models:
        df[f"{display_name}_response"] = ""

    total_questions = len(df)
    print(f"Processing {total_questions} questions with {len(models)} models...")
    print(f"Reading questions from '{question_column}' column")
    print("Models: gpt-4o-mini (OpenAI), deepseek-reasoner (DeepSeek), llama-4-maverick (OpenRouter)")
    print(f"Using {max_workers} parallel workers")
    print("=" * 80)

    logs = []
    lock = Lock()
    completed = 0
    failed_queries = []

    def process_single_query(args):
        nonlocal completed, logs, failed_queries

        idx, row, ontology_context = args
        query = row[question_column]  # Use the specified question column
        ontology_name = row["Root Entity"]
        answer_type = row["Answer Type"].strip().lower()

        # Prompt instructions
        if answer_type == "binary":
            instructions = "Only answer with TRUE or FALSE. Do not explain your answer."
        elif answer_type == "multi choice":
            instructions = (
                "Only answer with either:\n"
                "- A full individual's name (all lowercase with underscores, including birth year, e.g. caroline_lavinia_tubb_1840), or\n"
                "- A class starting with a capital letter.\n"
                "If there are multiple classes or individuals separate each by a comma and a space.\n"
                "Do not explain your answer. Only output the answer to the question."
            )
        else:
            instructions = ""

        full_prompt = f"{instructions}\n\nQuestion: {query}\nContext: {ontology_context}"
        query_results = {}
        query_failed = False

        for display_name, model_id in models.items():
            max_retries = 2  # Reduced retries
            content = None

            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    messages = [{"role": "user", "content": full_prompt}]
                    gen_params = model_params[display_name].copy()  # Make a copy to avoid modifying original

                    # Select appropriate client based on model
                    if display_name == "gpt-4o-mini":
                        client = openai_client
                        actual_model = model_id
                    elif display_name == "deepseek-reasoner":
                        client = deepseek_client
                        actual_model = model_id
                    elif display_name == "llama-4-maverick":
                        client = openrouter_client
                        actual_model = model_id
                    else:
                        raise ValueError(f"Unknown model: {display_name}")

                    response = client.chat.completions.create(
                        model=actual_model,
                        messages=messages,
                        **gen_params
                    )
                    content = response.choices[0].message.content
                    model_version = get_model_version_info(response, model_id)

                    end_time = time.time()
                    query_results[display_name] = content

                    # Log successful call
                    with lock:
                        log_entry = {
                            "Query_index": idx,
                            "Query": query,
                            "question_column_used": question_column,
                            "model": display_name,
                            "model_api_id": model_id,
                            "model_version": model_version,
                            "python_version": sys.version.split()[0],
                            "timestamp_request": datetime.now().isoformat(),
                            "response_time_sec": round(end_time - start_time, 3),
                            "response_preview": content[:100],
                            "attempt": attempt + 1,
                            "api_provider": "OpenAI" if display_name == "gpt-4o-mini" else
                            "DeepSeek" if display_name == "deepseek-reasoner" else "OpenRouter"
                        }
                        log_entry.update(model_params[display_name])
                        logs.append(log_entry)

                    break  # Success

                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        error_type = type(e).__name__
                        error_msg = str(e)

                        # Better error formatting
                        if hasattr(e, 'status_code'):
                            content = f"[ERROR] HTTP {e.status_code}: {error_msg}"
                        elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                            content = f"[ERROR] HTTP {e.response.status_code}: {error_msg}"
                        elif 'rate limit' in error_msg.lower():
                            content = f"[ERROR] Rate limit: {error_msg}"
                        else:
                            content = f"[ERROR] {error_type}: {error_msg}"

                        query_results[display_name] = content
                        query_failed = True

                        with lock:
                            if show_qa:
                                print(f"❌ Q{idx+1} - {display_name}: {content}")
                    else:
                        time.sleep(1)  # Brief wait before retry

        # Update DataFrame
        for display_name in models:
            df.at[idx, f"{display_name}_response"] = query_results.get(display_name, "[ERROR] No response")

        # Track progress
        with lock:
            completed += 1
            if query_failed:
                failed_queries.append(idx)

            if show_qa:
                print(f"\n✅ Q{idx+1}/{total_questions}: {query}")
                print(f"   Entity: {ontology_name} | Type: {answer_type}")
                for display_name in models:
                    result = query_results.get(display_name, "No response")
                    print(f"   🤖 {display_name}: {result}")
                print("-" * 60)
            elif completed % 10 == 0 or completed == total_questions:
                success_rate = ((completed - len(failed_queries)) / completed) * 100
                print(f"Progress: {completed}/{total_questions} ({success_rate:.1f}% success rate)")

        return idx, query_results

    # Pre-load all ontology contexts to avoid file I/O in threads
    print("Pre-loading ontology contexts...")
    ontology_contexts = {}

    for idx, row in df.iterrows():
        ontology_name = row["Root Entity"]

        if ontology_name not in ontology_contexts:
            if context_mode == "ttl":
                file_ext = ".ttl"
            elif context_mode == "json":
                file_ext = ".json"
            else:
                raise ValueError(f"Unknown context_mode: {context_mode}")

            ontology_path = os.path.join(ontology_base_path, f"{ontology_name}{file_ext}")

            try:
                if context_mode == "ttl":
                    with open(ontology_path, 'r', encoding='utf-8') as f:
                        ontology_contexts[ontology_name] = f.read()
                elif context_mode == "json":
                    import json
                    with open(ontology_path, 'r', encoding='utf-8') as f:
                        ontology_contexts[ontology_name] = json.dumps(json.load(f), indent=2)
            except FileNotFoundError:
                ontology_contexts[ontology_name] = f"[ERROR: {file_ext.upper()} file not found]"
                print(f"⚠️  Warning: Could not find {ontology_path}")

    # Process in parallel
    print(f"Starting parallel processing...")
    start_total = time.time()

    # Prepare arguments for all queries
    query_args = []
    for idx, row in df.iterrows():
        ontology_name = row["Root Entity"]
        ontology_context = ontology_contexts[ontology_name]
        query_args.append((idx, row, ontology_context))

    # Process all queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_query, args) for args in query_args]

        # Wait for completion
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Parallel processing error: {e}")

    end_total = time.time()
    total_time = end_total - start_total

    print("=" * 80)
    print(f"✅ COMPLETED!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average time per query: {total_time/total_questions:.2f}s")
    print(f"Successful queries: {completed - len(failed_queries)}/{completed}")
    print(f"Failed queries: {len(failed_queries)}")
    if failed_queries:
        print(f"Failed query indices: {failed_queries[:10]}{'...' if len(failed_queries) > 10 else ''}")

    return df, logs


def resume_failed_queries(df, failed_indices, ontology_base_path, models=MODELS, model_params=None, context_mode="ttl", question_column="Question"):
    """Resume processing only the failed queries"""
    if not failed_indices:
        print("No failed queries to resume.")
        return df, []

    print(f"Resuming {len(failed_indices)} failed queries...")
    failed_df = df.iloc[failed_indices].copy()
    failed_df = failed_df.reset_index(drop=True)

    # Run only failed queries with more conservative settings
    result_df, logs = run_llm_reasoning(
        failed_df,
        ontology_base_path,
        models=models,
        model_params=model_params,
        context_mode=context_mode,
        show_qa=True,
        max_workers=2,  # Slower for problem queries
        question_column=question_column
    )

    # Update original DataFrame
    for i, failed_idx in enumerate(failed_indices):
        for col in result_df.columns:
            if col.endswith('_response'):
                df.at[failed_idx, col] = result_df.iloc[i][col]

    return df, logs