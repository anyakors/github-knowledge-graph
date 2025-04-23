import spacy
import networkx as nx
from pathlib import Path
import json
import os
from git_fetch import parse_arguments, get_user_repos, get_readme, count_words

# Load NLP model - use a larger model for better entity recognition
nlp = spacy.load("en_core_web_lg")

# Define relevant concept categories for our knowledge graph
RELEVANT_CATEGORIES = [
    "TECH", "ORG", "PRODUCT", "ALGORITHM", "LANGUAGE", "FRAMEWORK", "LIBRARY", 
    "CONCEPT", "METHOD", "TOOL"
]

def extract_concepts(text):
    """Extract technical concepts from text using NLP"""
    doc = nlp(text)
    
    # Extract entities that might be technical concepts
    concepts = []
    
    # Get named entities
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            concepts.append({"text": ent.text, "type": ent.label_})
    
    # Get noun chunks that might be technical terms
    for chunk in doc.noun_chunks:
        # Filter for likely technical terms
        if any(token.pos_ == "PROPN" for token in chunk) or \
           any(token.text.lower() in ["algorithm", "framework", "library", "api", 
                                     "tool", "language", "model"] 
               for token in chunk):
            concepts.append({"text": chunk.text, "type": "TECH_TERM"})
    
    # Add custom rules for identifying programming languages, frameworks, etc.
    # This would need to be expanded with a comprehensive list
    tech_keywords = ["python", "javascript", "react", "tensorflow", "pytorch", 
                     "django", "flask", "machine learning", "deep learning",
                     "neural network", "algorithm", "data structure"]
    
    for keyword in tech_keywords:
        if keyword.lower() in text.lower():
            concepts.append({"text": keyword, "type": "TECH_TERM"})
    
    # Remove duplicates while preserving type
    unique_concepts = {}
    for concept in concepts:
        text = concept["text"].lower()
        if text not in unique_concepts:
            unique_concepts[text] = concept
    
    return list(unique_concepts.values())

def find_relationships(concepts, text):
    """Find relationships between concepts based on co-occurrence"""
    relationships = []
    
    # Create relationships based on co-occurrence within the same paragraph
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph_concepts = []
        for concept in concepts:
            if concept["text"].lower() in paragraph.lower():
                paragraph_concepts.append(concept)
        
        # Create relationships between all concepts in the same paragraph
        for i in range(len(paragraph_concepts)):
            for j in range(i+1, len(paragraph_concepts)):
                relationships.append({
                    "source": paragraph_concepts[i]["text"].lower(),
                    "target": paragraph_concepts[j]["text"].lower(),
                    "type": "CO_OCCURS_WITH"
                })
    
    return relationships

def update_knowledge_graph(graph, concepts, relationships, repo_name):
    """Update the knowledge graph with new concepts and relationships"""
    # Add concepts as nodes
    for concept in concepts:
        concept_id = concept["text"].lower()
        
        if concept_id not in graph.nodes:
            graph.add_node(concept_id, type=concept["type"], label=concept["text"], 
                          repositories=[repo_name])
        else:
            # Update existing node
            if repo_name not in graph.nodes[concept_id]["repositories"]:
                graph.nodes[concept_id]["repositories"].append(repo_name)
    
    # Add relationships as edges
    for rel in relationships:
        if rel["source"] in graph.nodes and rel["target"] in graph.nodes:
            if graph.has_edge(rel["source"], rel["target"]):
                # Increment weight of existing edge
                graph[rel["source"]][rel["target"]]["weight"] += 1
            else:
                # Create new edge with initial weight of 1
                graph.add_edge(rel["source"], rel["target"], 
                              type=rel["type"], weight=1)
    
    return graph

def save_graph(graph, output_path):
    """Save the knowledge graph to JSON format"""
    data = {
        "nodes": [],
        "edges": []
    }
    
    # Save nodes
    for node_id in graph.nodes:
        node_data = graph.nodes[node_id]
        data["nodes"].append({
            "id": node_id,
            "label": node_data["label"],
            "type": node_data["type"],
            "repositories": node_data["repositories"]
        })
    
    # Save edges
    for source, target in graph.edges:
        edge_data = graph[source][target]
        data["edges"].append({
            "source": source,
            "target": target,
            "type": edge_data["type"],
            "weight": edge_data["weight"]
        })
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_existing_graph(input_path):
    """Load an existing knowledge graph from JSON file"""
    if not os.path.exists(input_path):
        return nx.Graph()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    graph = nx.Graph()
    
    # Load nodes
    for node in data["nodes"]:
        graph.add_node(node["id"], 
                      label=node["label"], 
                      type=node["type"], 
                      repositories=node["repositories"])
    
    # Load edges
    for edge in data["edges"]:
        graph.add_edge(edge["source"], edge["target"], 
                      type=edge["type"], 
                      weight=edge["weight"])
    
    return graph

def main():
    args = parse_arguments()
    
    # Add argument for knowledge graph path
    knowledge_graph_path = args.output_dir + "/knowledge_graph.json"
    
    # Load existing knowledge graph if it exists
    graph = load_existing_graph(knowledge_graph_path)
    
    # Get GitHub token from args or environment
    github_token = args.token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Please provide a GitHub token using --token or set the GITHUB_TOKEN environment variable")
        return
    
    username = args.username
    output_dir = args.output_dir
    min_words = args.min_words
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all repositories for the user
    print(f"Fetching repositories for {username}...")
    repos = get_user_repos(username, github_token)
    print(f"Found {len(repos)} repositories")
    
    # Process each repository
    for repo in repos:
        repo_name = repo["name"]
        print(f"Processing repository: {repo_name}")
        
        readme = get_readme(username, repo_name, github_token)
        if readme:
            word_count = count_words(readme)
            if word_count >= min_words:
                # Save README file
                readme_path = f"{output_dir}/{repo_name}_README.md"
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(readme)
                
                # Extract concepts and relationships
                concepts = extract_concepts(readme)
                relationships = find_relationships(concepts, readme)
                
                # Update knowledge graph
                graph = update_knowledge_graph(graph, concepts, relationships, repo_name)
                
                print(f"Added {len(concepts)} concepts and {len(relationships)} relationships from {repo_name}")
    
    # Save updated knowledge graph
    save_graph(graph, knowledge_graph_path)
    print(f"Knowledge graph saved to {knowledge_graph_path}")
    print(f"Graph contains {len(graph.nodes)} concepts and {len(graph.edges)} relationships")

if __name__ == "__main__":
    main()