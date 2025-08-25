import pandas as pd
import os
from api_calls import run_llm_reasoning, resume_failed_queries

# Paths - using the new output directory
QUESTIONS_CSV = os.path.join("output", "llm_reasoning_results", "1hop_family_output_20250719_153248", "generated_questions.csv")
TTL_ONTOLOGY_DIR = os.path.join("src", "main", "resources", "ontologies", "family_1hop_tbox")

# Load the same CSV file
questions_path = QUESTIONS_CSV
sparql_df = pd.read_csv(questions_path)

print(f"Loaded {len(sparql_df)} SPARQL questions from the same CSV file")
print("Starting LLM reasoning with TTL ontologies...")
print("Using SPARQL queries from 'SPARQL Query' column")
print("Models: gpt-4o-mini (OpenAI), deepseek-reasoner (DeepSeek), llama-4-maverick (OpenRouter)")

# Run with optimized settings - using "SPARQL Query" column
ttl_results_df, logs = run_llm_reasoning(
    sparql_df,
    ontology_base_path=TTL_ONTOLOGY_DIR,
    context_mode="ttl",
    show_qa=True,   # Set to True if you want to see each Q&A in terminal
    max_workers=5,   # Adjust based on your API rate limits
    question_column="SPARQL Query"  # Specify the column to use
)

# Save results immediately
output_file = os.path.join("output", "llm_reasoning_results", "family_1hop_results_sparql_ttl.csv")
ttl_results_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to {output_file}")

# Check for any failures and optionally resume
failed_indices = []
model_columns = [col for col in ttl_results_df.columns if col.endswith('_response')]

for idx, row in ttl_results_df.iterrows():
    if any(str(row[col]).startswith('[ERROR]') for col in model_columns):
        failed_indices.append(idx)

if failed_indices:
    print(f"\n⚠️  {len(failed_indices)} queries had errors.")
    print("Failed query indices:", failed_indices[:20], "..." if len(failed_indices) > 20 else "")

    retry = input(f"Resume failed queries? This will cost additional API calls. (y/n): ")
    if retry.lower() == 'y':
        print("Resuming failed queries...")
        ttl_results_df, retry_logs = resume_failed_queries(
            ttl_results_df, failed_indices,
            TTL_ONTOLOGY_DIR,
            context_mode="ttl",
            question_column="SPARQL Query"
        )
        ttl_results_df.to_csv(output_file, index=False)
        print(f"✅ Resumed results saved to {output_file}")
else:
    print("🎉 All queries completed successfully!")

# Final summary
total_responses = len(ttl_results_df) * len(model_columns)
error_responses = 0
for idx, row in ttl_results_df.iterrows():
    error_responses += sum(1 for col in model_columns if str(row[col]).startswith('[ERROR]'))

success_rate = ((total_responses - error_responses) / total_responses) * 100
print(f"\nFinal Success Rate: {success_rate:.1f}% ({total_responses - error_responses}/{total_responses} responses)")