from nano_graphrag.graphrag import GraphRAG
from nano_graphrag._op import chunking_by_markdown_recursive
from nano_graphrag.base import QueryParam

# Example markdown text with nested structure
markdown_text = """
# Main Document Title

## Introduction
This is a comprehensive introduction to our topic.
It contains multiple sentences and paragraphs.

This is a separate paragraph in the introduction.

## Technical Details
### Architecture
The architecture consists of several components:
- Component A: Handles data input
  - Subcomponent A1: Preprocessing
  - Subcomponent A2: Validation
- Component B: Processes data
  - Subcomponent B1: Core processing
  - Subcomponent B2: Post-processing

### Implementation
The implementation follows these steps:
1. Initialize the system
2. Load configuration
   - Check environment variables
   - Validate settings
3. Start processing
   - Monitor progress
   - Handle errors

## Results
Our results show significant improvements:
* Metric 1 improved by 25%
* Metric 2 showed 30% better performance
* Metric 3 remained stable

### Detailed Analysis
The detailed analysis reveals several key findings that we need to consider carefully.
Each finding has its own implications for the project's future direction.

## Conclusion
In conclusion, we have demonstrated the effectiveness of our approach.
Future work will focus on further optimizations and improvements.
"""

# Initialize GraphRAG with the combined chunking method
rag = GraphRAG(
    working_dir="./workspace_combined",
    chunk_func=chunking_by_markdown_recursive,
    # Parameters for final chunks (RecursiveCharacterTextSplitter)
    chunk_token_size=512,
    chunk_overlap_token_size=50,
    # Additional parameters specific to the combined method
    chunk_func_params={
        "markdown_chunk_size": 1024,  # Size for initial markdown chunks
        "markdown_overlap": 100,      # Overlap for markdown chunks
        "recursive_separators": [
            "\n## ",     # Major sections
            "\n### ",    # Subsections
            "\n\n",      # Paragraphs
            "\n",        # Lines
            ". ",        # Sentences
            " ",         # Words
            ""          # Characters
        ]
    }
)

# Insert the document
rag.insert(markdown_text)

# Query to test the chunking
result = rag.query(
    "What are the main sections and their key points?",
    param=QueryParam(mode="local")
)
print("Results using combined Markdown and Recursive chunking:")
print(result)

# You can also query specific sections
section_query = rag.query(
    "What are the components of the architecture?",
    param=QueryParam(mode="local")
)
print("\nArchitecture components:")
print(section_query) 