# GitHub Knowledge Graph

A tool for building knowledge graphs from GitHub README files, focusing on software development, algorithms, computer science, and machine learning concepts.

## Overview

This project extracts technical concepts from GitHub README files and builds a graph of their relationships. It identifies programming languages, frameworks, algorithms, and other technical terms, and maps how they relate to each other based on co-occurrence and contextual patterns.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/github-knowledge-graph.git
   cd github-knowledge-graph
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_lg
   ```

3. Set up your GitHub token (required for API access):
   ```bash
   export GITHUB_TOKEN=your_github_token
   ```

## Usage

### Adding repositories from a GitHub user

```bash
python main.py add-user tensorflow --min-words 200
```

This will:
- Fetch all repositories from the specified GitHub user
- Extract README files with at least 200 words
- Identify technical concepts in the README files
- Add the concepts and their relationships to the knowledge graph

### Adding more repositories

You can add repositories from multiple users to build a more comprehensive graph:

```bash
python main.py add-user scikit-learn --min-words 200
```

### Visualizing the graph

Generate an interactive HTML visualization of the knowledge graph:

```bash
python main.py visualize
```

The visualization will be saved to `knowledge_graph/knowledge_graph.html` by default.

### Querying the graph

Search for information about a specific concept:

```bash
python main.py query "machine learning"
```

This will return information about the concept, including:
- Repositories where it appears
- Direct relationships with other concepts
- Related concepts within a specified distance in the graph

## Options

- `--min-words`: Minimum word count for README files (default: 100)
- `--token`: GitHub API token (can also be set via GITHUB_TOKEN environment variable)
- `--graph-dir`: Directory for storing the knowledge graph (default: knowledge_graph)
- `--distance`: Maximum distance in graph queries (default: 2)
- `--output`: Custom output path for visualizations

## Project Structure

```
github-knowledge-graph/
├── git_fetch.py              # Script for fetching README files
├── kgraph/
│   ├── concept_extractor.py  # Functions for extracting concepts
│   ├── relationship_extractor.py # Functions for relationship extraction
│   ├── graph_builder.py      # Functions for building and updating the graph
│   ├── graph_visualizer.py   # Functions for visualizing the graph
│   └── graph_query.py        # Functions for querying the graph
├── main.py                   # Main CLI script
├── requirements.txt          # Package dependencies
└── README.md                 # Project documentation
```

## License

[MIT License](LICENSE)
