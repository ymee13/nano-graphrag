# Local File Search with GraphRAG

This example demonstrates how to use nano-graphrag to implement a local file search system that can index and search through files in your local directory.

## Features

- Index text files (.txt, .md, .py, .js, .html, .css)
- Semantic search using sentence transformers
- Local embedding computation without requiring external API
- File metadata tracking

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Use the LocalFileSearch class as shown in the example:

```python
async def main():
    searcher = LocalFileSearch()
    # Index your documents directory
    await searcher.index_directory("./your_documents")
    # Search for relevant files
    results = await searcher.search("python async functions")
    for result in results:
        print(f"Document {result['id']}: {result['content'][:200]}...")
```

## Configuration

You can customize the following parameters:

- working_dir: Directory for storing GraphRAG data
- embedding_model: Change the sentence-transformer model
- top_k: Number of results to return in search

## Notes

- The implementation uses all-MiniLM-L6-v2 for embeddings which provides a good balance between performance and quality
- File indexing is done asynchronously for better performance
- Results include both the content and metadata about the source files