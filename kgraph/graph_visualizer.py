def visualize_knowledge_graph(graph_path, output_path=None):
    """Create a visualization of the knowledge graph using pyvis"""
    import json
    from pyvis.network import Network
    
    # Load the graph
    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a network
    net = Network(height="750px", width="100%", notebook=False, directed=False)
    
    # Define colors for different node types
    color_map = {
        "LANGUAGE": "#4CAF50",       # Green
        "FRAMEWORK_LIBRARY": "#2196F3", # Blue
        "ALGORITHM_DS": "#FFC107",   # Amber
        "ML_AI": "#9C27B0",          # Purple
        "DEV_CONCEPT": "#F44336",    # Red
        "TECH_TERM": "#607D8B",      # Blue Grey
        "ORG": "#FF5722",           # Deep Orange
        "PRODUCT": "#795548"         # Brown
    }
    
    # Add nodes
    for node in data["nodes"]:
        # Size based on number of repositories
        size = 10 + len(node["repositories"]) * 5
        color = color_map.get(node["type"], "#607D8B")  # Default to blue grey
        
        net.add_node(node["id"], 
                    label=node["label"], 
                    title=f"Type: {node['type']}<br>Repos: {', '.join(node['repositories'])}",
                    size=size,
                    color=color,
                    font={'size': 100, 'face': 'Arial', 'color': 'black'}, 
                    fixed=False)
    
    # Add edges
    for edge in data["edges"]:
        # Line width based on weight
        width = 1 + edge["weight"] * 0.5
        net.add_edge(edge["source"], edge["target"], 
                    value=width, 
                    title=f"Weight: {edge['weight']}")
    
    # Set physics layout
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=200)
    
    # Save the network
    if output_path is None:
        output_path = graph_path.replace(".json", ".html")
    
    net.save_graph(output_path)
    print(f"Visualization saved to {output_path}")