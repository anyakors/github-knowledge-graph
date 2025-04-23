import argparse
import os
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="GitHub Knowledge Graph Builder"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add user command
    add_parser = subparsers.add_parser("add-user", help="Add repositories from a GitHub user")
    add_parser.add_argument("username", help="GitHub username")
    add_parser.add_argument("--min-words", type=int, default=100, 
                          help="Minimum word count for README files (default: 100)")
    add_parser.add_argument("--token", help="GitHub token (optional if GITHUB_TOKEN env var is set)")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize the knowledge graph")
    viz_parser.add_argument("--output", help="Output HTML file path")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("term", help="Term to query")
    query_parser.add_argument("--distance", type=int, default=2, 
                            help="Maximum distance in the graph (default: 2)")
    
    # Common arguments
    parser.add_argument("--graph-dir", default="knowledge_graph", 
                      help="Directory for the knowledge graph (default: knowledge_graph)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create graph directory if it doesn't exist
    Path(args.graph_dir).mkdir(parents=True, exist_ok=True)
    
    graph_path = os.path.join(args.graph_dir, "knowledge_graph.json")
    
    if args.command == "add-user":
        # Import modules needed for adding a user
        from kgraph.concept_extractor import extract_tech_concepts, create_tech_concept_recognizer
        from kgraph.relationship_extractor import extract_relationships
        from kgraph.graph_builder import load_existing_graph, update_knowledge_graph, save_graph
        from git_fetch import get_user_repos, get_readme, count_words
        
        # Load or create graph
        graph = load_existing_graph(graph_path)
        
        # Get GitHub token
        github_token = args.token or os.environ.get("GITHUB_TOKEN")
        if not github_token:
            print("Please provide a GitHub token using --token or set the GITHUB_TOKEN environment variable")
            return
        
        # Get repositories
        repos = get_user_repos(args.username, github_token)
        print(f"Found {len(repos)} repositories for {args.username}")
        
        # Create tech concept recognizer
        tech_terms = create_tech_concept_recognizer()
        
        # Process each repository
        for repo in repos:
            repo_name = repo["name"]
            print(f"Processing repository: {repo_name}")
            
            readme = get_readme(args.username, repo_name, github_token)
            if readme:
                word_count = count_words(readme)
                if word_count >= args.min_words:
                    # Extract concepts and relationships
                    concepts = extract_tech_concepts(readme, tech_terms)
                    relationships = extract_relationships(concepts, readme)
                    
                    # Update knowledge graph
                    graph = update_knowledge_graph(graph, concepts, relationships, 
                                                 f"{args.username}/{repo_name}")
                    
                    print(f"Added {len(concepts)} concepts and {len(relationships)} relationships from {repo_name}")
        
        # Save graph
        save_graph(graph, graph_path)
        print(f"Knowledge graph updated and saved to {graph_path}")
        print(f"Graph now contains {len(graph.nodes)} concepts and {len(graph.edges)} relationships")
        
    elif args.command == "visualize":
        # Import visualization module
        from kgraph.graph_visualizer import visualize_knowledge_graph
        
        output_path = args.output if args.output else os.path.join(args.graph_dir, "knowledge_graph.html")
        visualize_knowledge_graph(graph_path, output_path)
        
    elif args.command == "query":
        # Import query module
        from kgraph.graph_query import query_knowledge_graph
        import json
        
        result = query_knowledge_graph(graph_path, args.term, args.distance)
        print(json.dumps(result, indent=2))
    
    else:
        print("Please specify a command: add-user, visualize, or query")

if __name__ == "__main__":
    main()