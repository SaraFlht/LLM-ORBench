import rdflib
import random
import argparse
from rdflib import URIRef

def copy_graph(g):
    new_g = rdflib.Graph()
    for triple in g:
        new_g.add(triple)
    return new_g

def add_triples_random(g_no_noise, noise_percentage):
    max_triples = int(noise_percentage * len(g_no_noise))
    noisy_g_random = rdflib.Graph()
    new_g_random = copy_graph(g_no_noise)
    num_triples = 0
    subjects = list(set(g_no_noise.subjects()))
    objects = list(set(g_no_noise.objects()))
    triples_list = list(g_no_noise)
    while num_triples < max_triples:
        triple = random.choice(triples_list)
        s, p, o = triple
        if random.choice([True, False]):
            new_s = random.choice(subjects)
            corrupted_triple = (new_s, p, o)
        else:
            new_o = random.choice(objects)
            corrupted_triple = (s, p, new_o)
        if corrupted_triple not in g_no_noise:
            noisy_g_random.add(corrupted_triple)
            new_g_random.add(corrupted_triple)
            num_triples += 1
    return noisy_g_random, new_g_random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ontology file (Turtle format)")
    parser.add_argument("--output", required=True, help="Output ontology file (Turtle format)")
    parser.add_argument("--noise_percentage", type=float, default=0.25, help="Fraction of triples to add as noise (0-1)")
    args = parser.parse_args()

    g = rdflib.Graph()
    g.parse(args.input, format="turtle")
    noisy_g, new_g = add_triples_random(g, args.noise_percentage)
    noisy_g.serialize(destination=args.output, format="turtle")
    print(f"Noisy ontology written to {args.output}") 