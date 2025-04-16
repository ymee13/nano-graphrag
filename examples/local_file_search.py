import os
from pathlib import Path
from typing import List, Dict, Any
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
import numpy as np
from sentence_transformers import SentenceTransformer

class LocalFileSearch:
    def __init__(self, working_dir: str = "./graphrag_data"):
        self.working_dir = working_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.graph_rag = self._initialize_graphrag()

    def _initialize_graphrag(self) -> GraphRAG:
        @wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
        async def local_embedding(texts: List[str]) -> np.ndarray:
            embeddings = self.model.encode(texts)
            return np.array(embeddings)

        return GraphRAG(
            working_dir=self.working_dir,
            embedding_func=local_embedding,
            embedding_batch_num=32,
            embedding_func_max_async=8
        )

    async def index_directory(self, directory: str) -> None:
        """Index all text files in the given directory"""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.txt', '.md', '.py', '.js', '.html', '.css')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            await self.graph_rag.add_text(
                                content,
                                metadata={
                                    'file_path': str(file_path),
                                    'file_name': file
                                }
                            )
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant files based on the query"""
        param = QueryParam(
            only_need_context=True,
            local_max_consider_chunk=top_k
        )
        results = await self.graph_rag.query(query, param=param)
        return self._parse_results(results)

    def _parse_results(self, results: str) -> List[Dict[str, Any]]:
        """Parse the results from GraphRAG into a structured format"""
        parsed_results = []
        # Basic parsing of the CSV-like output from GraphRAG
        lines = results.split('\n')
        for line in lines:
            if line and not line.startswith(('---', 'id', '```')):
                try:
                    id_content = line.split('\t')
                    if len(id_content) >= 2:
                        parsed_results.append({
                            'id': id_content[0],
                            'content': id_content[1]
                        })
                except Exception:
                    continue
        return parsed_results

# Example usage:
'''
async def main():
    searcher = LocalFileSearch()
    # Index a directory
    await searcher.index_directory("./your_documents")
    # Search for relevant files
    results = await searcher.search("python async functions")
    for result in results:
        print(f"Document {result['id']}: {result['content'][:200]}...")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''