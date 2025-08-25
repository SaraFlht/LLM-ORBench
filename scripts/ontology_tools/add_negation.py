import rdflib
import random
import argparse
from rdflib import URIRef

def get_object_properties(g):
    # Return all unique predicates that look like object properties (heuristic: start with 'is' or 'has')
    return set(p for s, p, o in g if isinstance(p, URIRef) and (str(p).startswith('is') or str(p).startswith('has')))

def create_negative_predicate(predicate):
    pred_str = str(predicate)
    if pred_str.startswith("is"):
        return URIRef(pred_str.replace("is", "isNot", 1))
    elif pred_str.startswith("has"):
        return URIRef(pred_str.replace("has", "hasNo", 1))
    else:
        return URIRef("not_" + pred_str)

def add_negation_triples(g, negation_percentage, contradiction_percentage=0.0):
    new_g = g.__class__()
    new_g += g
    all_triples = list(g)
    # Only consider triples with object properties
    object_properties = get_object_properties(g)
    op_triples = [triple for triple in all_triples if triple[1] in object_properties]
    num_negations = int(len(op_triples) * negation_percentage)
    num_contradictions = int(num_negations * contradiction_percentage)
    if num_negations == 0:
        return new_g
    negation_triples = random.sample(op_triples, num_negations)
    contradiction_triples = set(random.sample(negation_triples, num_contradictions)) if num_contradictions > 0 else set()

    for triple in negation_triples:
        s, p, o = triple
        neg_p = create_negative_predicate(p)
        new_g.add((s, neg_p, o))
        # If contradiction, positive already present, so nothing extra needed
    return new_g

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ontology file (Turtle format)")
    parser.add_argument("--output", required=True, help="Output ontology file")
    parser.add_argument("--negation_percentage", type=float, default=0.25, help="Fraction of triples to add negation for (0-1)")
    parser.add_argument("--contradiction_percentage", type=float, default=0.0, help="Fraction of negations that are contradictions (0-1)")
    args = parser.parse_args()

    g = rdflib.Graph()
    g.parse(args.input, format="turtle")
    new_g = add_negation_triples(g, args.negation_percentage, args.contradiction_percentage)
    new_g.serialize(destination=args.output, format="turtle")
    print(f"Negation ontology written to {args.output}") 