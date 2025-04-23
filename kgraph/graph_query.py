def query_knowledge_graph(graph_path, query_term, max_distance=2):
    """Query the knowledge graph for concepts related to the query term"""
    import json
    import networkx as nx
    
    # Load the graph
    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node in data["nodes"]:
        G.add_node(node["id"], **node)
    
    # Add edges
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"], **edge)
    
    # Find the closest match to the query term
    query_lower = query_term.lower()
    best_match = None
    best_score = 0
    
    for node_id in G.nodes:
        if query_lower in node_id:
            similarity = len(query_lower) / len(node_id)
            if similarity > best_score:
                best_score = similarity
                best_match = node_id
    
    if best_match is None:
        return {"error": f"No concept found matching '{query_term}'"}
    
    # Get all nodes within max_distance
    related_nodes = {}
    for node in nx.single_source_shortest_path_length(G, best_match, cutoff=max_distance):
        if node != best_match:
            distance = nx.shortest_path_length(G, best_match, node)
            related_nodes[node] = {
                "label": G.nodes[node]["label"],
                "type": G.nodes[node]["type"],
                "distance": distance,
                "repositories": G.nodes[node]["repositories"]
            }
    
    # Get direct relationships
    direct_relationships = []
    for neighbor in G.neighbors(best_match):
        edge_data = G[best_match][neighbor]
        direct_relationships.append({
            "concept": G.nodes[neighbor]["label"],
            "type": edge_data["type"],
            "weight": edge_data["weight"]
        })
    
    result = {
        "query": query_term,
        "matched_concept": G.nodes[best_match]["label"],
        "type": G.nodes[best_match]["type"],
        "found_in_repos": G.nodes[best_match]["repositories"],
        "direct_relationships": direct_relationships,
        "related_concepts": related_nodes
    }
    
    return result