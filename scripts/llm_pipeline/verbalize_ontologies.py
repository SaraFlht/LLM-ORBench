"""
verbalize_ontologies.py
---------------------
Verbalize all ontologies in the family_1hop_tbox directory.
Converts TTL ontology files to natural language descriptions and saves as JSON.
Runnable from IDE without terminal.
"""
# Import necessary libraries
import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL
from pathlib import Path
import os
import re
import json
import glob
import inflect
p = inflect.engine()

def get_nice_label(g, entity):
    """Get a nice label for an entity."""
    # First check if there's a literal label
    label = g.value(entity, RDFS.label)
    if label:
        return str(label)

    # If no label, use the entity name/IRI
    if isinstance(entity, URIRef):
        uri_str = str(entity)
        if '#' in uri_str:
            name = uri_str.split('#')[-1]
        else:
            name = uri_str.split('/')[-1]

        # Convert camelCase or snake_case to space-separated words
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)  # Split camelCase
        words = words.replace('_', ' ')  # Replace underscores with spaces

        return words.capitalize()

    # For blank nodes, return None
    return None

def describe_class(g, cls, classes, subclass_relations, equivalent_class_relations, obj_properties, property_domains, property_ranges):
    """Generate a detailed natural language description of a class."""
    descriptions = []
    cls_name = get_nice_label(g, cls)

    # Subclass relationships
    parents = [o for s, o in subclass_relations if s == cls]
    for parent in parents:
        parent_name = get_nice_label(g, parent)
        descriptions.append(f"{cls_name} is a subclass of {parent_name}.")

    # Equivalent class relationships
    equivalents = [o for s, o in equivalent_class_relations if s == cls]
    for eq in equivalents:
        if isinstance(eq, URIRef):
            eq_name = get_nice_label(g, eq)
            descriptions.append(f"{cls_name} is equivalent to {eq_name}.")
        else:
            # For blank nodes, try to interpret the structure
            union_members = list(g.triples((eq, OWL.unionOf, None)))
            intersection_members = list(g.triples((eq, OWL.intersectionOf, None)))

            if union_members:
                union_list = union_members[0][2]
                members = []
                while union_list != RDF.nil:
                    member = g.value(union_list, RDF.first)
                    if member:
                        member_name = get_nice_label(g, member)
                        if member_name:  # Skip blank nodes
                            members.append(member_name)
                    union_list = g.value(union_list, RDF.rest)

                if members:
                    descriptions.append(f"{cls_name} is defined as the union of {', '.join(members)}.")

            elif intersection_members:
                intersection_list = intersection_members[0][2]
                members = []
                while intersection_list != RDF.nil:
                    member = g.value(intersection_list, RDF.first)
                    if member:
                        member_name = get_nice_label(g, member)
                        if member_name:  # Skip blank nodes
                            members.append(member_name)
                    intersection_list = g.value(intersection_list, RDF.rest)

                if members:
                    descriptions.append(f"{cls_name} is defined as the intersection of {', '.join(members)}.")

    # Properties that have this class as domain
    domain_props = []
    for prop in obj_properties:
        domains = property_domains.get(prop, set())
        if cls in domains:
            domain_props.append(prop)

    if domain_props:
        for prop in domain_props:
            prop_name = get_nice_label(g, prop)
            ranges = property_ranges.get(prop, set())
            range_names = [get_nice_label(g, r) for r in ranges if get_nice_label(g, r)]  # Skip blank nodes

            if range_names:
                descriptions.append(f"The domain of {prop_name} is {cls_name}, which means that {cls_name} can {prop_name} {', '.join(range_names)}.")
            else:
                descriptions.append(f"The domain of {prop_name} is {cls_name}, which means that {cls_name} can {prop_name} other things.")

    # Properties that have this class as range
    range_props = []
    for prop in obj_properties:
        ranges = property_ranges.get(prop, set())
        if cls in ranges:
            range_props.append(prop)

    if range_props:
        for prop in range_props:
            prop_name = get_nice_label(g, prop)
            domains = property_domains.get(prop, set())
            domain_names = [get_nice_label(g, d) for d in domains if get_nice_label(g, d)]  # Skip blank nodes

            if domain_names:
                descriptions.append(f"The range of {prop_name} is {cls_name}, which means that {', '.join(domain_names)} can {prop_name} {cls_name}.")
            else:
                descriptions.append(f"The range of {prop_name} is {cls_name}, which means that other things can {prop_name} {cls_name}.")

    # Add a final statement about the class type
    descriptions.append(f"{cls_name} is defined as a class in this ontology.")

    return " ".join(descriptions)

def describe_property(g, prop, obj_properties, property_domains, property_ranges):
    """Generate a detailed natural language description of a property."""
    descriptions = []
    prop_name = get_nice_label(g, prop)

    # Check if it's functional, symmetric, transitive, etc.
    is_functional = False
    is_symmetric = False
    is_transitive = False
    is_inverse_functional = False

    for s, p, o in g.triples((prop, RDF.type, None)):
        if o == OWL.FunctionalProperty:
            is_functional = True
        elif o == OWL.SymmetricProperty:
            is_symmetric = True
        elif o == OWL.TransitiveProperty:
            is_transitive = True
        elif o == OWL.InverseFunctionalProperty:
            is_inverse_functional = True

    # Describe basic property type
    descriptions.append(f"{prop_name} is an object property.")

    # Add special characteristics
    if is_functional:
        descriptions.append(f"It is a functional property, meaning each subject can have at most one {prop_name} relationship.")

    if is_symmetric:
        descriptions.append(f"It is a symmetric property, meaning if A {prop_name} B, then B {prop_name} A.")

    if is_transitive:
        descriptions.append(f"It is a transitive property, meaning if A {prop_name} B and B {prop_name} C, then A {prop_name} C.")

    if is_inverse_functional:
        descriptions.append(f"It is an inverse functional property, meaning each object can have at most one subject that {prop_name} it.")

    # Domain and range
    domains = property_domains.get(prop, set())
    ranges = property_ranges.get(prop, set())

    domain_names = [get_nice_label(g, d) for d in domains if get_nice_label(g, d)]  # Skip blank nodes
    domain_names = [name for name in domain_names if name is not None]  # Filter out None
    range_names = [get_nice_label(g, r) for r in ranges if get_nice_label(g, r)]  # Skip blank nodes
    range_names = [name for name in range_names if name is not None]  # Filter out None

    if domain_names:
        domain_names = [name for name in domain_names if isinstance(name, str)]  # Ensure only strings
        descriptions.append(f"The domain of {prop_name} is {', '.join(domain_names)}, which means that only {' or '.join(domain_names)} can have this property.")

    if range_names:
        range_names = [name for name in range_names if isinstance(name, str)]  # Ensure only strings
        descriptions.append(f"The range of {prop_name} is {', '.join(range_names)}, which means this property can only point to {' or '.join(range_names)}.")

    # Check for inverse properties
    inverse_props = []
    for s, p, o in g.triples((prop, OWL.inverseOf, None)):
        if isinstance(o, URIRef):  # Skip blank nodes
            inverse_props.append(o)
    for s, p, o in g.triples((None, OWL.inverseOf, prop)):
        if isinstance(s, URIRef):  # Skip blank nodes
            inverse_props.append(s)

    if inverse_props:
        inv_names = [get_nice_label(g, inv) for inv in inverse_props if get_nice_label(g, inv)]  # Skip blank nodes
        if inv_names:
            inv_names = [name for name in inv_names if isinstance(name, str)]  # Ensure only strings
            if inv_names:
                descriptions.append(f"The inverse of {prop_name} is {', '.join(inv_names)}.")

    # Check for sub-properties
    sub_props = []
    for s, p, o in g.triples((None, RDFS.subPropertyOf, prop)):
        if isinstance(s, URIRef):  # Skip blank nodes
            sub_props.append(s)

    if sub_props:
        sub_names = [get_nice_label(g, sub) for sub in sub_props if get_nice_label(g, sub)]  # Skip blank nodes
        if sub_names:
            sub_names = [name for name in sub_names if isinstance(name, str)]  # Ensure only strings
            if sub_names:
                descriptions.append(f"The sub-properties of {prop_name} include {', '.join(sub_names)}.")

    # Check for parent properties
    parent_props = []
    for s, p, o in g.triples((prop, RDFS.subPropertyOf, None)):
        if isinstance(o, URIRef):  # Skip blank nodes
            parent_props.append(o)

    if parent_props:
        parent_names = [get_nice_label(g, parent) for parent in parent_props if get_nice_label(g, parent)]  # Skip blank nodes
        if parent_names:
            parent_names = [name for name in parent_names if isinstance(name, str)]  # Ensure only strings
            if parent_names:
                descriptions.append(f"{prop_name} is a sub-property of {', '.join(parent_names)}.")

    # Check for property chains
    chains = []
    for s, p, o in g.triples((prop, OWL.propertyChainAxiom, None)):
        chain_list = o
        chain_props = []
        while chain_list != RDF.nil:
            chain_prop = g.value(chain_list, RDF.first)
            if chain_prop and isinstance(chain_prop, URIRef):  # Skip blank nodes
                chain_prop_name = get_nice_label(g, chain_prop)
                if chain_prop_name:
                    chain_props.append(chain_prop_name)
            chain_list = g.value(chain_list, RDF.rest)

        if chain_props:
            chain_description = " followed by ".join(chain_props)
            chains.append(chain_description)

    if chains:
        descriptions.append(f"{prop_name} is defined as a chain of: {'; '.join(chains)}.")

    return " ".join(descriptions)

def describe_individual(g, ind, classes, obj_properties):
    """Generate a detailed natural language description of an individual."""
    descriptions = []
    ind_name = get_nice_label(g, ind)

    # Get the types of the individual
    types = []
    for s, p, o in g.triples((ind, RDF.type, None)):
        if o != OWL.NamedIndividual and o in classes:
            types.append(o)

    if types:
        type_names = [get_nice_label(g, t) for t in types if get_nice_label(g, t)]  # Skip blank nodes
        type_names = [name for name in type_names if name is not None]  # Filter out None
        if type_names:
            type_names = [name for name in type_names if isinstance(name, str)]  # Ensure only strings
            if type_names:
                descriptions.append(f"{ind_name} is a {', '.join(type_names)}.")
        else:
            descriptions.append(f"{ind_name} is an individual in this ontology.")
    else:
        descriptions.append(f"{ind_name} is an individual in this ontology.")

    # Get outgoing relationships
    outgoing_relations = []
    for s, p, o in g.triples((ind, None, None)):
        if p != RDF.type and isinstance(o, URIRef) and p in obj_properties:
            outgoing_relations.append((p, o))

    if outgoing_relations:
        for prop, obj in outgoing_relations:
            prop_name = get_nice_label(g, prop)
            obj_name = get_nice_label(g, obj)
            if prop_name and obj_name:  # Skip blank nodes
                descriptions.append(f"{ind_name} {prop_name} {obj_name}.")

    # Get incoming relationships
    incoming_relations = []
    for s, p, o in g.triples((None, None, ind)):
        if isinstance(s, URIRef) and s != ind and p in obj_properties:
            incoming_relations.append((s, p))

    if incoming_relations:
        for subj, prop in incoming_relations:
            subj_name = get_nice_label(g, subj)
            prop_name = get_nice_label(g, prop)
            if subj_name and prop_name:  # Skip blank nodes
                descriptions.append(f"{subj_name} {prop_name} {ind_name}.")

    # Get annotation properties
    annotations = []
    for s, p, o in g.triples((ind, None, None)):
        if isinstance(o, Literal):
            annotations.append((p, o))

    if annotations:
        for prop, value in annotations:
            prop_name = get_nice_label(g, prop)
            if prop_name:  # Skip blank nodes
                descriptions.append(f"{ind_name} has {prop_name} {value}.")

    return " ".join(descriptions)

def verbalize_ontology(ttl_file_path, output_dir):
    """Verbalize a single ontology file and save as JSON."""
    print(f"Processing: {ttl_file_path}")
    
    # Load the ontology with RDFLib
    g = Graph()
    try:
        g.parse(ttl_file_path, format="turtle")
        print(f"Successfully loaded ontology with {len(g)} triples")
    except Exception as e:
        print(f"Error loading ontology: {e}")
        return None

    # Extract ontology structure
    classes = set()
    for s, p, o in g.triples((None, RDF.type, OWL.Class)):
        if isinstance(s, URIRef):
            classes.add(s)

    obj_properties = set()
    for s, p, o in g.triples((None, RDF.type, OWL.ObjectProperty)):
        if isinstance(s, URIRef):
            obj_properties.add(s)

    data_properties = set()
    for s, p, o in g.triples((None, RDF.type, OWL.DatatypeProperty)):
        if isinstance(s, URIRef):
            data_properties.add(s)

    # Only include true individuals (rdf:type owl:NamedIndividual)
    individuals = set()
    for s, p, o in g.triples((None, RDF.type, OWL.NamedIndividual)):
        if isinstance(s, URIRef):
            individuals.add(s)

    # Optional fallback for legacy ontologies (if no individuals found)
    if not individuals:
        for s, p, o in g.triples((None, RDF.type, None)):
            if (
                o != OWL.Class and
                o != OWL.ObjectProperty and
                o != OWL.DatatypeProperty and
                o != OWL.AnnotationProperty and
                o != OWL.Ontology and
                isinstance(s, URIRef) and
                not str(s).endswith('.owl')
            ):
                individuals.add(s)

    subclass_relations = []
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        if s in classes and o in classes:
            subclass_relations.append((s, o))

    equivalent_class_relations = []
    for s, p, o in g.triples((None, OWL.equivalentClass, None)):
        if s in classes:
            equivalent_class_relations.append((s, o))

    # Get domain and range for properties
    property_domains = {}
    property_ranges = {}

    for prop in obj_properties:
        domains = set()
        for s, p, o in g.triples((prop, RDFS.domain, None)):
            domains.add(o)
        property_domains[prop] = domains

        ranges = set()
        for s, p, o in g.triples((prop, RDFS.range, None)):
            ranges.add(o)
        property_ranges[prop] = ranges

    # Create verbalization data
    verbalization_data = {
        "ontology": {
            "name": os.path.basename(ttl_file_path).split('.')[0],
            "triples": len(g)
        },
        "classes": [],
        "objectProperties": [],
        "dataProperties": [],
        "individuals": []
    }

    # Add classes to JSON
    for cls in sorted(classes, key=lambda x: get_nice_label(g, x) or ""):
        cls_name = get_nice_label(g, cls)
        if cls_name:  # Skip blank nodes
            cls_data = {
                "uri": str(cls),
                "classLabel": cls_name,
                "description": describe_class(g, cls, classes, subclass_relations, equivalent_class_relations, obj_properties, property_domains, property_ranges)
            }
            verbalization_data["classes"].append(cls_data)

    # Add object properties to JSON
    for prop in sorted(obj_properties, key=lambda x: get_nice_label(g, x) or ""):
        prop_name = get_nice_label(g, prop)
        if prop_name:  # Skip blank nodes
            prop_data = {
                "uri": str(prop),
                "propertyLabel": prop_name,
                "description": describe_property(g, prop, obj_properties, property_domains, property_ranges)
            }
            verbalization_data["objectProperties"].append(prop_data)

    # Add data properties to JSON
    for prop in sorted(data_properties, key=lambda x: get_nice_label(g, x) or ""):
        prop_name = get_nice_label(g, prop)
        if prop_name:  # Skip blank nodes
            prop_data = {
                "uri": str(prop),
                "propertyLabel": prop_name,
                "description": f"{prop_name} is a data property in this ontology."
            }
            verbalization_data["dataProperties"].append(prop_data)

    # Add individuals to JSON
    for ind in sorted(individuals, key=lambda x: get_nice_label(g, x) or ""):
        ind_name = get_nice_label(g, ind)
        if ind_name:  # Skip blank nodes
            ind_data = {
                "uri": str(ind),
                "individualLabel": ind_name,
                "description": describe_individual(g, ind, classes, obj_properties)
            }
            verbalization_data["individuals"].append(ind_data)

    # Save JSON file
    output_file = output_dir / f"{os.path.basename(ttl_file_path).split('.')[0]}.json"
    with open(output_file, "w") as json_file:
        json.dump(verbalization_data, json_file, indent=2)
    
    print(f"Saved verbalization to: {output_file}")
    return output_file

def main():
    """Main function to process all ontologies in family_1hop_tbox."""
    # Create output directories
    output_dir = Path("verbalized_ontologies/family_2hop")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all TTL files in family_1hop_tbox
    ttl_files = glob.glob("../src/main/resources/ontologies/family_2hop_tbox/*.ttl")
    
    if not ttl_files:
        print("No TTL files found in ../src/main/resources/ontologies/family_2hop_tbox/")
        return

    print(f"Found {len(ttl_files)} TTL files to process")
    
    processed_count = 0
    for ttl_file in ttl_files:
        try:
            result = verbalize_ontology(ttl_file, output_dir)
            if result:
                processed_count += 1
            # (Removed: triple-based verbalization call)
        except Exception as e:
            print(f"Error processing {ttl_file}: {e}")
    
    print(f"\nProcessed {processed_count} out of {len(ttl_files)} files successfully!")
    print(f"Results saved in: {output_dir.absolute()}")

if __name__ == "__main__":
    main() 