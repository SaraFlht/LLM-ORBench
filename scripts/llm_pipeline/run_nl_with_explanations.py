"""
Natural Language + Explanations: Tests LLM reasoning with human-readable questions and explanation context
Uses explanations directly from CSV instead of separate ontology files
"""
import pandas as pd
import os
import time
import json
import gc
import psutil
from pathlib import Path
from datetime import datetime
from api_calls import (
    run_llm_reasoning,
    log_models_metadata,
    calculate_model_performance_summary,
    check_api_clients,
    openai_client,
    deepseek_client,
    openrouter_client
)

# Navigate to project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
os.chdir(project_root)

print(f"Working directory set to: {os.getcwd()}")

def monitor_memory():
    """Monitor system memory usage"""
    return psutil.virtual_memory().percent

def force_cleanup():
    """Force garbage collection"""
    gc.collect()
    time.sleep(1)

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"output/llm_results/explanations_experiment_{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"📂 Output directory: {output_dir}")

# Configuration for NL + Explanations experiment
MODELS_CONFIG = {
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "deepseek-chat": "deepseek-chat",
    "llama-4-maverick": "meta-llama/llama-4-maverick"
}

CONFIG = {
    'experiment_type': 'nl_with_explanations',
    'description': 'Natural language questions with explanation context from CSV',
    'questions_csv': "output/FamilyOWL/1hop/test.csv",  # CSV with explanations column
    'context_mode': 'explanations',  # Use explanations instead of ontology files
    'question_column': 'Question',
    'explanation_column': 'Explanation',  # Column containing explanations
    'models_used': MODELS_CONFIG,
    'max_workers': 8,
    'batch_size': 25,
    'checkpoint_frequency': 50,
    'silent_mode': False,
    'test_mode': False,  # Set to True for testing with fewer questions
    'memory_threshold': 85
}

def print_experiment_header():
    """Print a nice experiment header"""
    print("\n" + "="*80)
    print("🚀 NATURAL LANGUAGE + EXPLANATIONS EXPERIMENT")
    print("="*80)
    print(f"📋 Experiment: {CONFIG['description']}")
    print(f"🤖 Models: {', '.join(CONFIG['models_used'].keys())}")
    print(f"🔧 Max Workers: {CONFIG['max_workers']}")
    print(f"📦 Batch Size: {CONFIG['batch_size']}")
    print(f"💾 Checkpoint: Every {CONFIG['checkpoint_frequency']} questions")
    print(f"🔇 Silent Mode: {'ON' if CONFIG['silent_mode'] else 'OFF'}")
    print(f"🧠 Memory Threshold: {CONFIG['memory_threshold']}%")
    print(f"📂 Output: {output_dir}")
    print("="*80)

def validate_setup():
    """Validate experiment setup with memory monitoring"""
    print("🔍 Validating setup...")
    print(f"🧠 Initial memory usage: {monitor_memory():.1f}%")

    # Load questions with explanations
    try:
        df = pd.read_csv(CONFIG['questions_csv'])
        print(f"✅ Loaded {len(df)} questions from {CONFIG['questions_csv']}")
    except FileNotFoundError:
        print(f"❌ Error: Questions CSV not found at {CONFIG['questions_csv']}")
        return None

    # Check if explanation column exists
    if CONFIG['explanation_column'] not in df.columns:
        print(f"❌ Error: Explanation column '{CONFIG['explanation_column']}' not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return None

    # Analyze explanations
    explanations_available = df[CONFIG['explanation_column']].notna().sum()
    total_questions = len(df)

    print(f"📊 Explanation Analysis:")
    print(f"   Total questions: {total_questions}")
    print(f"   Questions with explanations: {explanations_available} ({explanations_available/total_questions*100:.1f}%)")

    if explanations_available == 0:
        print("❌ Error: No explanations found in the specified column")
        return None

    # Analyze explanation lengths
    explanation_lengths = df[CONFIG['explanation_column']].dropna().str.len()
    print(f"   Avg explanation length: {explanation_lengths.mean():.1f} characters")
    print(f"   Min/Max explanation length: {explanation_lengths.min()}/{explanation_lengths.max()} characters")

    return df

def analyze_explanation_patterns(df):
    """Analyze patterns in explanations"""
    explanation_col = CONFIG['explanation_column']

    patterns = {
        'total_explanations': 0,
        'short_explanations': 0,  # < 100 chars
        'medium_explanations': 0,  # 100-500 chars
        'long_explanations': 0,   # > 500 chars
        'step_based_explanations': 0,  # Contains numbered steps
        'reasoning_keywords': 0,  # Contains reasoning words
    }

    reasoning_keywords = ['because', 'therefore', 'since', 'thus', 'shows', 'indicates', 'implies', 'reason']

    for idx, row in df.iterrows():
        explanation = str(row.get(explanation_col, ''))

        if pd.notna(row[explanation_col]) and explanation.strip():
            patterns['total_explanations'] += 1

            # Length analysis
            length = len(explanation)
            if length < 100:
                patterns['short_explanations'] += 1
            elif length <= 500:
                patterns['medium_explanations'] += 1
            else:
                patterns['long_explanations'] += 1

            # Pattern analysis
            if any(f"{i}." in explanation or f"{i})" in explanation for i in range(1, 6)):
                patterns['step_based_explanations'] += 1

            if any(keyword in explanation.lower() for keyword in reasoning_keywords):
                patterns['reasoning_keywords'] += 1

    return patterns

def estimate_experiment_time(df, models_config, max_workers):
    """Estimate experiment duration"""
    estimated_time_per_question = 2.5  # Slightly faster since no file loading
    total_calls = len(df) * len(models_config)
    estimated_total_time = (total_calls * estimated_time_per_question) / max_workers

    print(f"\n⏱️ EXPERIMENT ESTIMATES:")
    print(f"   Total questions: {len(df)}")
    print(f"   Total API calls: {total_calls}")
    print(f"   Estimated time: {estimated_total_time/60:.1f} minutes")
    return estimated_total_time

def check_for_previous_run():
    """Check if there's a previous incomplete run"""
    recovery_file = output_dir / "LATEST_recovery_info.json"
    if recovery_file.exists():
        try:
            with open(recovery_file, 'r') as f:
                recovery_info = json.load(f)

            print(f"\n🔄 Found previous incomplete run:")
            print(f"   Completed: {recovery_info['questions_completed']}/{recovery_info['total_questions']} questions")
            print(f"   Progress: {recovery_info['completion_percentage']:.1f}%")

            response = input("Do you want to continue from where you left off? (y/n): ").lower().strip()
            if response == 'y':
                latest_csv = output_dir / "LATEST_checkpoint.csv"
                if latest_csv.exists():
                    df = pd.read_csv(latest_csv)
                    return df, recovery_info

        except Exception as e:
            print(f"⚠️ Could not load previous run: {e}")

    return None, None

def create_explanation_context(row, explanation_col):
    """Create explanation context from CSV row"""
    explanation = row.get(explanation_col, '')

    if pd.isna(explanation) or not str(explanation).strip():
        return "No specific explanation available for this reasoning step."

    return str(explanation).strip()

def main():
    """Main experiment execution"""
    print_experiment_header()

    if not check_api_clients():
        print("❌ Fix API client issues before proceeding")
        return

    # Log model metadata
    models_metadata_log = log_models_metadata(
        MODELS_CONFIG,
        output_dir,
        openai_client,
        deepseek_client,
        openrouter_client
    )

    # Check for previous run
    previous_df, recovery_info = check_for_previous_run()

    if previous_df is not None:
        df = previous_df
        print(f"✅ Resuming from previous run with {len(df)} questions")
        start_question = recovery_info['questions_completed']
    else:
        # Validate setup for new run
        df = validate_setup()
        if df is None:
            return
        start_question = 0

    # Check initial memory
    initial_memory = monitor_memory()
    if initial_memory > 80:
        print(f"⚠️ Warning: High initial memory usage ({initial_memory:.1f}%)")

    # Apply test mode if enabled
    if CONFIG['test_mode']:
        df_sample = df.head(50).copy()
        print(f"🧪 TEST MODE: Using {len(df_sample)} questions")
    else:
        df_sample = df.copy()
        print(f"🏭 PRODUCTION MODE: Processing all {len(df_sample)} questions")

    # Analyze explanation patterns
    explanation_patterns = analyze_explanation_patterns(df_sample)
    print(f"\n📊 Explanation Pattern Analysis:")
    for pattern, count in explanation_patterns.items():
        if explanation_patterns['total_explanations'] > 0:
            percentage = (count / explanation_patterns['total_explanations']) * 100
            print(f"   {pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

    # Prepare data for explanation-enhanced reasoning
    # We need to create a mock ontology context since the original function expects file-based context
    print(f"\n🔧 Preparing explanation-enhanced context...")

    # Create a temporary directory structure for the explanation "ontologies"
    temp_explanation_dir = output_dir / "temp_explanations"
    temp_explanation_dir.mkdir(exist_ok=True)

    # Create individual explanation files for each question
    for idx, row in df_sample.iterrows():
        explanation_context = create_explanation_context(row, CONFIG['explanation_column'])

        # Use question index as the "entity" name for file lookup
        entity_name = f"question_{idx}"
        explanation_file = temp_explanation_dir / f"{entity_name}.json"

        # Create explanation as JSON context
        explanation_data = {
            "question": row[CONFIG['question_column']],
            "explanation": explanation_context,
            "answer": row.get('Answer', 'Unknown'),
            "context_type": "reasoning_explanation"
        }

        with open(explanation_file, 'w') as f:
            json.dump(explanation_data, f, indent=2)

        # Update the dataframe to use our entity naming
        df_sample.at[idx, 'Root Entity'] = entity_name

    # Estimate time
    estimated_time = estimate_experiment_time(df_sample, MODELS_CONFIG, CONFIG['max_workers'])

    # Confirmation prompt
    print(f"\n⚠️ Ready to start experiment")
    print(f"📊 Processing {len(df_sample)} questions across {len(MODELS_CONFIG)} models")
    print(f"🎯 Starting from question {start_question + 1}")
    print(f"🔇 Silent mode: {'ON (faster)' if CONFIG['silent_mode'] else 'OFF (shows responses)'}")
    print(f"🧠 Current memory: {monitor_memory():.1f}%")

    # Auto-start after brief pause
    print("Starting in 3 seconds... (Ctrl+C to cancel)")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n❌ Experiment cancelled by user")
        return

    # Run experiment
    print(f"\n🧠 Starting explanation-enhanced reasoning experiment...")
    start_time = time.time()

    try:
        results_df, logs, detailed_metrics, _ = run_llm_reasoning(
            df_sample,
            ontology_base_path=str(temp_explanation_dir),  # Use our temp explanation dir
            models=CONFIG['models_used'],
            context_mode='json',  # Use JSON mode for explanations
            max_workers=CONFIG['max_workers'],
            question_column=CONFIG['question_column'],
            batch_size=CONFIG['batch_size'],
            output_dir=output_dir,
            silent_mode=CONFIG['silent_mode']
        )

        experiment_time = time.time() - start_time

        # Print completion summary
        print_completion_summary(results_df, experiment_time, estimated_time, explanation_patterns)

        # Save final results
        save_final_results(results_df, logs, detailed_metrics, experiment_time, explanation_patterns)

    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        experiment_time = time.time() - start_time
        print(f"⏱️ Ran for {experiment_time:.1f}s ({experiment_time/60:.1f}min)")
        print(f"💾 Progress has been saved in checkpoints")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup temp directory
        import shutil
        if temp_explanation_dir.exists():
            shutil.rmtree(temp_explanation_dir)
            print(f"🗑️ Cleaned up temporary explanation files")

def print_completion_summary(results_df, actual_time, estimated_time, explanation_patterns):
    """Print experiment completion summary"""
    print("\n" + "="*80)
    print("🎉 EXPLANATION-ENHANCED EXPERIMENT COMPLETED!")
    print("="*80)

    # Time analysis
    print(f"⏱️ Time Analysis:")
    print(f"   Actual time: {actual_time:.1f}s ({actual_time/60:.1f}min)")
    print(f"   Estimated time: {estimated_time:.1f}s ({estimated_time/60:.1f}min)")
    time_diff = ((actual_time - estimated_time) / estimated_time) * 100
    print(f"   Difference: {time_diff:+.1f}%")

    # Memory analysis
    final_memory = monitor_memory()
    print(f"🧠 Memory Analysis:")
    print(f"   Final memory usage: {final_memory:.1f}%")

    # Response analysis
    total_questions = len(results_df)
    print(f"\n📊 Response Analysis:")
    print(f"   Total questions: {total_questions}")

    for model in MODELS_CONFIG.keys():
        response_col = f"{model}_response"
        if response_col in results_df.columns:
            responses = results_df[response_col]
            non_empty = (responses != "").sum()
            errors = responses.astype(str).str.startswith('[ERROR]').sum()
            success = non_empty - errors
            success_rate = (success / len(responses)) * 100 if len(responses) > 0 else 0

            # Correctness analysis
            correctness_col = f"{model}_quality_correctness"
            if correctness_col in results_df.columns:
                correct_answers = (results_df[correctness_col] > 0.5).sum()
                correctness_rate = (correct_answers / len(results_df)) * 100
                avg_correctness = results_df[correctness_col].mean()
                avg_response_time = pd.to_numeric(results_df[f"{model}_response_time"], errors='coerce').mean()

                print(f"   🤖 {model}:")
                print(f"     Completed: {success}/{len(responses)} ({success_rate:.1f}%)")
                print(f"     Correct: {correct_answers}/{len(results_df)} ({correctness_rate:.1f}%)")
                print(f"     Avg correctness: {avg_correctness:.3f}")
                print(f"     Avg response time: {avg_response_time:.2f}s")
                if errors > 0:
                    print(f"     Errors: {errors}")

def save_final_results(results_df, logs, detailed_metrics, experiment_time, explanation_patterns):
    """Save all final results with enhanced metadata"""
    print(f"\n💾 Saving final results...")

    # Main results files
    results_file = output_dir / "explanation_enhanced_results_FINAL.csv"
    logs_file = output_dir / "explanation_enhanced_logs_FINAL.csv"
    metrics_file = output_dir / "explanation_enhanced_metrics_FINAL.json"
    config_file = output_dir / "experiment_summary.json"

    # Save main files
    results_df.to_csv(results_file, index=False)
    pd.DataFrame(logs).to_csv(logs_file, index=False)

    # Save essential metrics
    essential_metrics = []
    for metric in detailed_metrics:
        essential_metric = {
            "query_index": metric.get("query_index"),
            "model_display_name": metric.get("model_display_name"),
            "final_answer_extracted": metric.get("final_answer_extracted"),
            "quality_correctness": metric.get("quality_correctness"),
            "response_time_seconds": metric.get("response_time_seconds"),
            "error_occurred": metric.get("error_occurred")
        }
        essential_metrics.append(essential_metric)

    with open(metrics_file, 'w') as f:
        json.dump(essential_metrics, f, indent=2, default=str)

    # Enhanced experiment summary
    performance_summary = calculate_model_performance_summary(results_df, MODELS_CONFIG)
    experiment_summary = {
        'config': CONFIG,
        'explanation_patterns': explanation_patterns,
        'experiment_time_seconds': experiment_time,
        'experiment_time_minutes': experiment_time / 60,
        'total_questions_processed': len(results_df),
        'total_api_calls': len(results_df) * len(MODELS_CONFIG),
        'performance_summary': performance_summary,
        'timestamp': datetime.now().isoformat(),
        'output_directory': str(output_dir),
        'memory_info': {
            'final_memory_usage_percent': monitor_memory(),
            'memory_threshold': CONFIG['memory_threshold']
        },
        'files_created': {
            'results': str(results_file),
            'logs': str(logs_file),
            'metrics': str(metrics_file),
            'summary': str(config_file)
        }
    }

    with open(config_file, 'w') as f:
        json.dump(experiment_summary, f, indent=2, default=str)

    print(f"✅ Results saved to: {output_dir}")
    print(f"📄 Main results: {results_file}")
    print(f"📊 Summary: {config_file}")
    print(f"💾 Checkpoints: {output_dir / 'checkpoints'}")

    # Print final model performance summary
    print(f"\n📈 MODEL PERFORMANCE SUMMARY:")
    for model_name, summary in performance_summary.items():
        response_metrics = summary.get('response_metrics', {})
        quality_metrics = summary.get('quality_metrics', {})

        success_rate = response_metrics.get('success_rate', 0) * 100
        avg_correctness = quality_metrics.get('correctness', {}).get('mean', 0)

        print(f"  🤖 {model_name}:")
        print(f"     Success Rate: {success_rate:.1f}%")
        print(f"     Avg Correctness: {avg_correctness:.3f}")

    # Final cleanup
    force_cleanup()

if __name__ == "__main__":
    main()