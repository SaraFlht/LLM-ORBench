#!/usr/bin/env python3
"""
Wrapper script to run negation generation from IntelliJ.
Modify the parameters below as needed.
"""

import sys
import os

# Add the scripts directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from add_negation import add_negation_triples
import rdflib
from pathlib import Path

# Navigate to project root (2 levels up from current script)
script_dir = Path(__file__).resolve().parent  # scripts/llm_pipeline/
project_root = script_dir.parent.parent       # owl-inference-explainer/
os.chdir(project_root)

print(f"Working directory set to: {os.getcwd()}")

def main():
    # Configuration - modify these parameters as needed
    input_file = "../src/main/resources/ontologies/family_1hop_tbox/Person_john_william_folland.ttl"
    output_file = "../src/main/resources/ontologies/negation_output.ttl"
    negation_percentage = 0.25
    contradiction_percentage = 0.1
    
    print(f"Adding {negation_percentage * 100}% negation to {input_file}")
    print(f"Contradiction percentage: {contradiction_percentage * 100}%")
    print(f"Output will be saved to {output_file}")
    
    # Load the ontology
    g = rdflib.Graph()
    g.parse(input_file, format="turtle")
    print(f"Loaded ontology with {len(g)} triples")
    
    # Add negation
    new_g = add_negation_triples(g, negation_percentage, contradiction_percentage)
    
    # Save the negation-augmented ontology
    new_g.serialize(destination=output_file, format="turtle")
    print(f"Negation-augmented ontology saved with {len(new_g)} triples")
    print(f"Added {len(new_g) - len(g)} negation triples")

if __name__ == "__main__":
    main() 