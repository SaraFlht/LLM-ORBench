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

def get_longest_tag_explanation(explanations):
    """
    Find the explanation with the longest tag from a list of explanations.
    Returns the explanation without the TAG part.
    """
    if not explanations:
        return []

    max_tag_length = 0
    longest_explanation = []

    for explanation in explanations:
        tag_length = extract_tag_length(explanation)
        if tag_length > max_tag_length:
            max_tag_length = tag_length
            # Return explanation without the TAG part (all elements except the last one)
            longest_explanation = explanation[:-1] if len(explanation) > 1 else explanation

    return longest_explanation

def get_hardest_explanations_for_entry(matching_entries):
    """
    Get the hardest explanation from each matching entry.
    Returns a list of explanations (one from each entry).
    """
    hardest_explanations = []

    for entry in matching_entries:
        explanations = entry.get('explanations', [])
        longest_explanation = get_longest_tag_explanation(explanations)
        if longest_explanation:
            hardest_explanations.append(longest_explanation)

    return hardest_explanations

def process_files(csv_file_path, json_file_path, output_csv_path):
    """
    Process CSV and JSON files to extract the hardest explanations for each SPARQL query and Task ID.

    Args:
        csv_file_path (str): Path to the CSV file containing SPARQL queries and Task IDs
        json_file_path (str): Path to the JSON file containing explanations
        output_csv_path (str): Path to save the output CSV file

    Returns:
        pandas.DataFrame: Updated DataFrame with hardest explanations column
    """

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Read the JSON file
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Create mappings from SPARQL queries and Task IDs to their corresponding entries
    sparql_to_entries = defaultdict(list)
    task_id_to_entries = defaultdict(list)

    # Iterate through JSON data to build the mappings
    for key, value in json_data.items():
        # Map SPARQL queries
        if 'sparqlQueries' in value:
            for sparql_query in value['sparqlQueries']:
                sparql_to_entries[sparql_query].append(value)

        # Map Task IDs
        if 'taskIds' in value:
            for task_id in value['taskIds']:
                task_id_to_entries[task_id].append(value)

    # Initialize the hardest explanation column
    df['Hardest Explanation'] = ''

    # Process each row in the CSV
    for index, row in df.iterrows():
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
            # Get hardest explanations from all matching entries
            hardest_explanations = get_hardest_explanations_for_entry(matching_entries)

            # Convert to string format for CSV storage
            explanations_str = ' , '.join([str(explanation) for explanation in hardest_explanations])
            df.at[index, 'Hardest Explanation'] = explanations_str
        else:
            df.at[index, 'Hardest Explanation'] = ''
            print(f"Warning: No matching entries found for row {index}:")
            print(f"  SPARQL Query: {sparql_query}")
            print(f"  Task ID: {task_id}")

    # Save the updated CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to: {output_csv_path}")

    return df

# File paths
input_csv = 'output/OWL2Bench/2hop/SPARQL_questions.csv'
input_json = 'output/OWL2Bench/2hop/Explanations.json'
output_csv = 'output/OWL2Bench/2hop/SPARQL_questions_with_hardest_explanations.csv'

# Process the files
result_df = process_files(input_csv, input_json, output_csv)

# Display the results
print("\nFirst few rows of the updated DataFrame:")
columns_to_show = ['SPARQL Query', 'Task ID', 'Hardest Explanation']
available_columns = [col for col in columns_to_show if col in result_df.columns]
print(result_df[available_columns].head())

# Show summary statistics
print(f"\nSummary:")
print(f"Total rows processed: {len(result_df)}")
print(f"Rows with hardest explanations: {(result_df['Hardest Explanation'] != '').sum()}")
print(f"Rows without hardest explanations: {(result_df['Hardest Explanation'] == '').sum()}")

# Display a sample of hardest explanations
print(f"\nSample hardest explanations:")
for i, explanation in enumerate(result_df['Hardest Explanation'].head(3)):
    if explanation:
        print(f"Row {i}: {explanation}")