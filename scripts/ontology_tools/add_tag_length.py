import pandas as pd
import json
import re
from collections import defaultdict
import os
from pathlib import Path

# Navigate to project root (2 levels up from current script)
script_dir = Path(__file__).resolve().parent  # scripts/ontology_tools/
project_root = script_dir.parent.parent       # LLM-ORBench/
os.chdir(project_root)

def extract_tag_length(explanation):
    """
    Extract the tag length from an explanation array.
    The tag is in the last element and follows the format "TAG:XXXXX"
    """
    if not explanation:
        return 0

    last_element = explanation[-1]
    # Look for TAG: pattern and extract what comes after it
    tag_match = re.search(r'TAG:([^"]*)', last_element)

    if tag_match:
        tag_content = tag_match.group(1).strip()
        # Remove any special characters that might be at the end and count actual tag characters
        # Keep letters, numbers, and common symbols like ∩
        clean_tag = re.sub(r'[^\w∩∪⊆⊇⊂⊃∀∃¬∧∨→↔≡≠≤≥∈∉⊊⊋]', '', tag_content)
        return len(clean_tag)

    return 0

def calculate_explanation_tag_stats(explanations):
    """
    Calculate min and max tag lengths from explanations array.
    """
    if not explanations:
        return 0, 0

    tag_lengths = []
    for explanation in explanations:
        tag_length = extract_tag_length(explanation)
        if tag_length > 0:  # Only count valid tags
            tag_lengths.append(tag_length)

    if not tag_lengths:
        return 0, 0

    return min(tag_lengths), max(tag_lengths)

def process_files(csv_file_path, json_file_path, output_csv_path):
    """
    Process CSV and JSON files to extract SPARQL query information and calculate tag sizes
    using both SPARQL Query and Task ID for matching.

    Args:
        csv_file_path (str): Path to the CSV file containing SPARQL queries and Task IDs
        json_file_path (str): Path to the JSON file containing explanations
        output_csv_path (str): Path to save the output CSV file

    Returns:
        pandas.DataFrame: Updated DataFrame with shortest and longest tag columns
    """

    # Read the CSV file
    print(f"Reading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    print(f"CSV loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Read the JSON file
    print(f"Reading JSON file: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print(f"JSON loaded successfully. Number of entries: {len(json_data)}")

    # Create mappings from SPARQL queries and Task IDs to their corresponding entries
    sparql_to_entries = defaultdict(list)
    task_id_to_entries = defaultdict(list)

    # Iterate through JSON data to build the mappings
    print("Building mappings...")
    for key, value in json_data.items():
        # Map SPARQL queries
        if 'sparqlQueries' in value:
            for sparql_query in value['sparqlQueries']:
                sparql_to_entries[sparql_query].append(value)

        # Map Task IDs
        if 'taskIds' in value:
            for task_id in value['taskIds']:
                task_id_to_entries[task_id].append(value)

    print(f"Mappings built. SPARQL queries: {len(sparql_to_entries)}, Task IDs: {len(task_id_to_entries)}")

    # Initialize/overwrite the columns
    df['Min Tag Length'] = 0
    df['Max Tag Length'] = 0

    # Process each row in the CSV
    print("Processing CSV rows...")
    for index, row in df.iterrows():
        if index % 1000 == 0:  # Progress indicator
            print(f"Processing row {index}/{len(df)}")

        sparql_query = row.get('SPARQL Query', '')
        task_id = row.get('Task ID', '')

        matching_entries = []

        # Find entries that match both SPARQL query and Task ID (if both are present)
        if sparql_query and task_id:
            # Get entries that contain both the SPARQL query and the Task ID
            sparql_entries = set(id(entry) for entry in sparql_to_entries.get(sparql_query, []))
            task_entries = set(id(entry) for entry in task_id_to_entries.get(task_id, []))

            # Find intersection - entries that have both the SPARQL query and Task ID
            common_entry_ids = sparql_entries.intersection(task_entries)

            # Get the actual entry objects
            for entry in sparql_to_entries.get(sparql_query, []):
                if id(entry) in common_entry_ids:
                    matching_entries.append(entry)

        elif sparql_query:
            # Fall back to SPARQL query only if Task ID is not available
            matching_entries = sparql_to_entries.get(sparql_query, [])

        elif task_id:
            # Fall back to Task ID only if SPARQL query is not available
            matching_entries = task_id_to_entries.get(task_id, [])

        if matching_entries:
            total_shortest = 0
            total_longest = 0

            # Calculate tag stats for each matching entry
            for entry in matching_entries:
                min_tag, max_tag = calculate_explanation_tag_stats(entry.get('explanations', []))
                total_shortest += min_tag
                total_longest += max_tag

            df.at[index, 'Min Tag Length'] = total_shortest
            df.at[index, 'Max Tag Length'] = total_longest
        else:
            # No matching entries found
            df.at[index, 'Min Tag Length'] = 0
            df.at[index, 'Max Tag Length'] = 0
            if index < 10:  # Only show first 10 warnings to avoid spam
                print(f"Warning: No matching entries found for row {index}:")
                print(f"  SPARQL Query: {sparql_query}")
                print(f"  Task ID: {task_id}")

    # Save the updated CSV
    print(f"Saving results to: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to: {output_csv_path}")

    return df

# File paths
input_csv = './output/OWL2Bench/1hop/SPARQL_questions.csv'
input_json = './output/OWL2Bench/1hop/Explanations.json'
output_csv = './output/OWL2Bench/1hop/SPARQL_questions_with_tags.csv'

# Check if files exist
if not os.path.exists(input_csv):
    print(f"Error: CSV file not found: {input_csv}")
    exit(1)
if not os.path.exists(input_json):
    print(f"Error: JSON file not found: {input_json}")
    exit(1)

# Process the files
try:
    result_df = process_files(input_csv, input_json, output_csv)

    # Display the results
    print("\nFirst few rows of the updated DataFrame:")
    columns_to_show = ['SPARQL Query', 'Task ID', 'Min Tag Length', 'Max Tag Length']
    available_columns = [col for col in columns_to_show if col in result_df.columns]
    print(result_df[available_columns].head())

    # Show summary statistics
    print(f"\nSummary:")
    print(f"Total rows processed: {len(result_df)}")
    print(f"Rows with tag data: {(result_df['Min Tag Length'] > 0).sum()}")
    print(f"Rows without tag data: {(result_df['Min Tag Length'] == 0).sum()}")

    # Display some statistics about the tag lengths
    print(f"\nTag Length Statistics:")
    print(f"Average Min Tag Length: {result_df['Min Tag Length'].mean():.2f}")
    print(f"Average Max Tag Length: {result_df['Max Tag Length'].mean():.2f}")
    print(f"Overall Min Tag Length: {result_df['Min Tag Length'].min()}")
    print(f"Overall Max Tag Length: {result_df['Max Tag Length'].max()}")

except Exception as e:
    print(f"An error occurred: {e}")