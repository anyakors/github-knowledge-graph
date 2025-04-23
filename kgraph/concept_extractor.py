def create_tech_concept_recognizer():
    """Create a specialized recognizer for technical concepts"""
    # Common programming languages
    programming_languages = [
        "python", "javascript", "typescript", "java", "c++", "c#", "ruby", "go", 
        "rust", "php", "swift", "kotlin", "scala", "haskell", "perl", "r", "julia"
    ]
    
    # Common frameworks and libraries
    frameworks_libraries = [
        # ML/Data Science
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", 
        "matplotlib", "seaborn", "huggingface", "transformers", "jax", "opencv",
        
        # Frontend
        "react", "angular", "vue", "svelte", "next.js", "nuxt", "gatsby", 
        "ember", "backbone", "jquery", "bootstrap", "tailwind", "material-ui", 
        "chakra-ui", "styled-components", "redux", "mobx", "recoil", "zustand",
        "d3.js", "three.js", "webpack", "vite", "rollup", "parcel",
        
        # Backend
        "django", "flask", "fastapi", "express", "nest.js", "spring", "spring boot",
        "laravel", "rails", "symfony", "aspnet", "phoenix", "koa", "hapi", 
        "rocket", "actix", "gin", "echo", "fiber", "grpc", "graphql", "apollo",
        
        # Mobile/Desktop
        "flutter", "react native", "swift ui", "jetpack compose", "xamarin",
        "electron", "tauri", "qt", "gtk", "maui",
        
        # DevOps/Infrastructure
        "kubernetes", "docker", "terraform", "pulumi", "ansible", "chef", "puppet",
        "nginx", "apache", "traefik", "prometheus", "grafana", "istio", "envoy"
    ]

    # Common algorithms and data structures
    algorithms_data_structures = [
        # Basic Data Structures
        "array", "linked list", "stack", "queue", "deque", "hash table", "hash map",
        "dictionary", "set", "tree", "binary tree", "binary search tree", "avl tree",
        "red-black tree", "b-tree", "b+ tree", "trie", "prefix tree", "suffix tree",
        "heap", "priority queue", "graph", "directed graph", "undirected graph",
        "weighted graph", "adjacency list", "adjacency matrix", "incidence matrix",
        "union-find", "disjoint set", "bloom filter", "lru cache",
        
        # Searching Algorithms
        "binary search", "linear search", "interpolation search", "exponential search",
        "fibonacci search", "jump search", "breadth-first search", "bfs", 
        "depth-first search", "dfs", "best-first search", "a* search algorithm",
        "dijkstra's algorithm", "bellman-ford algorithm", "floyd-warshall algorithm",
        "bidirectional search", "binary space partitioning",
        
        # Sorting Algorithms
        "bubble sort", "selection sort", "insertion sort", "merge sort", "quick sort",
        "heap sort", "counting sort", "radix sort", "bucket sort", "shell sort",
        "tim sort", "topological sort", "external sorting", "patience sort",
        
        # String Algorithms
        "string matching", "rabin-karp algorithm", "knuth-morris-pratt algorithm",
        "boyer-moore algorithm", "aho-corasick algorithm", "regular expression",
        "levenshtein distance", "longest common subsequence", "longest common substring",
        "suffix array", "burrows-wheeler transform", "run-length encoding",
        "huffman coding", "lempel-ziv-welch", "lzw compression",
        
        # Graph Algorithms
        "minimum spanning tree", "kruskal's algorithm", "prim's algorithm",
        "shortest path problem", "strongly connected components", "tarjan's algorithm",
        "kosaraju's algorithm", "bipartite graph", "maximum flow", "ford-fulkerson algorithm",
        "edmonds-karp algorithm", "dinic's algorithm", "push-relabel algorithm",
        "minimum cut", "maximum bipartite matching", "hopcroft-karp algorithm",
        "travelling salesman problem", "hamiltonian path", "eulerian path",
        "page rank", "isomorphism", "planarity testing",
        
        # Dynamic Programming
        "dynamic programming", "memoization", "fibonacci sequence", "knapsack problem",
        "longest increasing subsequence", "edit distance", "matrix chain multiplication",
        "optimal binary search tree", "longest palindromic subsequence",
        "coin change problem", "subset sum problem", "rod cutting problem",
        
        # Greedy Algorithms
        "greedy algorithm", "activity selection problem", "huffman coding",
        "fractional knapsack problem", "job sequencing", "interval scheduling",
        "minimum spanning tree", "dijkstra's algorithm",
        
        # Backtracking
        "backtracking", "n-queens problem", "sudoku solver", "hamiltonian cycle",
        "graph coloring", "rat in a maze", "subset sum", "permutations", "combinations",
        
        # Divide and Conquer
        "divide and conquer", "merge sort", "quick sort", "binary search",
        "strassen's matrix multiplication", "closest pair of points",
        "convex hull", "karatsuba algorithm", "fast fourier transform", "fft",
        
        # Computational Geometry
        "convex hull", "graham scan", "line intersection", "point in polygon",
        "closest pair of points", "voronoi diagram", "delaunay triangulation",
        "sweep line algorithm", "segment intersection", "bounding volume",
        
        # Randomized Algorithms
        "randomized algorithm", "quickselect", "randomized quicksort",
        "reservoir sampling", "monte carlo algorithm", "las vegas algorithm",
        "random walk", "skip list", "primality testing", "miller-rabin",
        
        # Bit Manipulation
        "bit manipulation", "bit masking", "bitwise operation", "bit shift",
        "bit counting", "bitboard", "xor swap", "gray code", "hamming distance",
        
        # Advanced Data Structures
        "segment tree", "fenwick tree", "binary indexed tree", "sparse table",
        "suffix array", "suffix tree", "suffix automaton", "van emde boas tree",
        "merkle tree", "quad tree", "octree", "kd-tree", "r-tree",
        "skip list", "treap", "splay tree", "rope", "persistent data structure",
        
        # Hashing Techniques
        "hash function", "collision resolution", "separate chaining",
        "open addressing", "linear probing", "quadratic probing", "double hashing",
        "perfect hashing", "cuckoo hashing", "consistent hashing", "locality-sensitive hashing",
        
        # Complexity Classes
        "time complexity", "space complexity", "big o notation", "big omega notation",
        "big theta notation", "p vs np", "np-complete", "np-hard", "polynomial time",
        "exponential time", "logarithmic time", "constant time", "amortized analysis"
    ]
    #algorithms_data_structures = [
    #    "binary search", "quicksort", "merge sort", "depth-first search", 
    #    "breadth-first search", "dynamic programming", "greedy algorithm",
    #    "backtracking", "binary tree", "linked list", "hash table", "stack", 
    #    "queue", "heap", "graph", "array", "sorting", "searching"
    #]
    
    # Common ML/AI concepts
    ml_ai_concepts = [
        # Core ML/AI Paradigms
        "machine learning", "deep learning", "artificial intelligence", 
        "neural network", "natural language processing", "computer vision",
        "reinforcement learning", "supervised learning", "unsupervised learning",
        "semi-supervised learning", "self-supervised learning", "transfer learning",
        "few-shot learning", "zero-shot learning", "one-shot learning",
        "federated learning", "active learning", "online learning",
        "representation learning", "meta-learning", "multi-task learning",
        
        # Machine Learning Tasks
        "classification", "regression", "clustering", "dimensionality reduction",
        "anomaly detection", "outlier detection", "time series analysis",
        "forecasting", "recommendation system", "ranking", "similarity search",
        "information retrieval", "sequence labeling", "pattern recognition",
        "object detection", "image segmentation", "named entity recognition",
        "sentiment analysis", "topic modeling", "text summarization",
        "question answering", "machine translation", "speech recognition",
        "speech synthesis", "image generation", "video generation",
        
        # Machine Learning Models and Algorithms
        "linear regression", "logistic regression", "decision tree", "random forest",
        "gradient boosting", "xgboost", "lightgbm", "catboost", "support vector machine",
        "naive bayes", "k-nearest neighbors", "k-means", "dbscan", "hierarchical clustering",
        "principal component analysis", "singular value decomposition", "t-sne", "umap",
        
        # Neural Network Architectures and Components
        "multilayer perceptron", "convolutional neural network", "cnn",
        "recurrent neural network", "rnn", "long short-term memory", "lstm",
        "gated recurrent unit", "gru", "transformer", "attention mechanism",
        "self-attention", "encoder-decoder", "autoencoder", "variational autoencoder",
        "generative adversarial network", "gan", "diffusion model", "graph neural network",
        "siamese network", "u-net", "residual network", "resnet", "inception", "efficientnet",
        "vision transformer", "vit", "bert", "gpt", "t5", "llama", "stable diffusion",
        
        # Training Concepts and Techniques
        "backpropagation", "gradient descent", "stochastic gradient descent", "sgd",
        "adam optimizer", "learning rate", "batch size", "epoch", "overfitting",
        "underfitting", "regularization", "dropout", "batch normalization",
        "layer normalization", "weight decay", "early stopping", "hyperparameter",
        "hyperparameter tuning", "cross-validation", "train-test split",
        "data augmentation", "fine-tuning", "prompting", "prompt engineering",
        "chain-of-thought", "retrieval-augmented generation", "rag",
        
        # Evaluation Metrics and Concepts
        "accuracy", "precision", "recall", "f1-score", "area under curve", "auc",
        "receiver operating characteristic", "roc", "confusion matrix",
        "mean squared error", "mse", "root mean squared error", "rmse",
        "mean absolute error", "mae", "r-squared", "silhouette score",
        "perplexity", "bleu score", "rouge score", "human evaluation",
        
        # AI Ethics and Responsible AI
        "bias", "fairness", "interpretability", "explainability", "transparency",
        "accountability", "privacy", "data privacy", "differential privacy",
        "federated learning", "model cards", "data cards", "responsible ai",
        "ethical ai", "ai alignment", "ai safety"
    ]
    
    # Software development concepts
    dev_concepts = [
        "api", "rest", "graphql", "microservice", "devops", "ci/cd", "git", 
        "docker", "kubernetes", "containerization", "serverless", "cloud", 
        "aws", "azure", "gcp", "testing", "unit test", "integration test",
        "agile", "scrum", "database", "sql", "nosql", "orm", "cache", "redis"
    ]
    
    # Combine all terms with their categories
    all_terms = []
    all_terms.extend([(term, "LANGUAGE") for term in programming_languages])
    all_terms.extend([(term, "FRAMEWORK_LIBRARY") for term in frameworks_libraries])
    all_terms.extend([(term, "ALGORITHM_DS") for term in algorithms_data_structures])
    all_terms.extend([(term, "ML_AI") for term in ml_ai_concepts])
    all_terms.extend([(term, "DEV_CONCEPT") for term in dev_concepts])
    
    return all_terms

def extract_tech_concepts(text, tech_terms):
    """Extract technical concepts using the list of known terms"""
    concepts = []
    text_lower = text.lower()
    
    for term, category in tech_terms:
        if term.lower() in text_lower:
            # Find all occurrences
            start = 0
            while True:
                start = text_lower.find(term.lower(), start)
                if start == -1:
                    break
                
                # Get the actual casing from the original text
                actual_term = text[start:start + len(term)]
                concepts.append({"text": actual_term, "type": category})
                start += len(term)
    
    return concepts