#!/usr/bin/env python3
"""
Wrapper script to run noise generation from IntelliJ.
Modify the parameters below as needed.
"""

import sys
import os

# Add the scripts directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from add_noise import add_triples_random
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
    output_file = "../src/main/resources/ontologies/noise_output.ttl"
    noise_percentage = 0.25
    
    print(f"Adding {noise_percentage * 100}% noise to {input_file}")
    print(f"Output will be saved to {output_file}")
    
    # Load the ontology
    g = rdflib.Graph()
    g.parse(input_file, format="turtle")
    print(f"Loaded ontology with {len(g)} triples")
    
    # Add noise
    noisy_g, new_g = add_triples_random(g, noise_percentage)
    
    # Save the noisy ontology
    noisy_g.serialize(destination=output_file, format="turtle")
    print(f"Noisy ontology saved with {len(noisy_g)} triples")
    print(f"Added {len(noisy_g) - len(g)} noise triples")

if __name__ == "__main__":
    main() 