"""
Simple FAISS-based tool retrieval system.
Retrieves relevant tools based on query similarity instead of passing all tool documentation.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import faiss
import dotenv
dotenv.load_dotenv()
class ToolRetriever:
    """
    Simple FAISS-based tool retrieval system.
    """
    def __init__(self, embeddings_path: str = None, top_k: int = 5, llm_engine: Any = None):
        """
        Initialize the tool retriever. 
        Args:
            embeddings_path: Path to the tool embeddings JSON file
            top_k: Number of top tools to retrieve
            llm_engine: The LLM engine to use
        """
        self.top_k = top_k
        self.tool_names = []
        self.tool_embeddings = None
        self.faiss_index = None
        self.toolbox_metadata = {}
        self.llm_engine = llm_engine
        # Default embeddings path
        if embeddings_path is None:
            embeddings_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "agents", 
                "embeddings", 
                "tool_embeddings.json"
            )       
        self.embeddings_path = embeddings_path
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load tool embeddings from JSON file."""
        with open(self.embeddings_path, 'r') as f:
            data = json.load(f)
        
        self.tool_names = list(data.keys())
        embeddings_list = []
        
        for tool_name in self.tool_names:
            embeddings_list.append(data[tool_name])
        
        self.tool_embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.tool_embeddings)
        
        # Create FAISS index
        dimension = self.tool_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(self.tool_embeddings)
        print(f"Loaded {len(self.tool_names)} tools into FAISS index")

    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query using OpenAI's text-embedding-3-large model.
        
        Args:
            query: The query string to embed
            
        Returns:
            Query embedding as numpy array
        """
        try:
            return self.llm_engine.embed_with_normalization(query)
        except Exception as e:
            print(f"Error embedding query with normalization: {str(e)}")
            return None
    
    def expand_query(self, query: str) -> str:
        """
        Expand the query for better tool retrieval.
        Focuses on actions and tool types rather than topic keywords.
        
        Args:
            query: Original user query
            
        Returns:
            Expanded query string
        """
        expansion_prompt = f"""You expand user queries specifically for embedding-based TOOL RETRIEVAL.
GOAL
- Produce ONE single-line ASCII, lowercase string that maximizes match to tool routing texts.
- Focus on ACTIONS and TOOL TYPES needed to solve the task (e.g., paper search, pdf fetch, pdf reader, paper summarizer, paper qa).
- Prefer tool-type tokens and short functionality phrases over topic keywords.
- Positive-only phrasing; no negations.
- Include: task intent; modality (pdf/web/image/text if known); concrete tool types and what each does; key IO hints (title/url/doi/pdf_path); constraints (language, printed vs handwritten, etc.) ONLY if relevant.
- Keep subject-matter nouns to a minimum; do NOT list general buzzwords.
- Return exactly the expanded line (nothing else, no explanation, no markdown, no backticks).
Expand the following query: {query}"""
        response = self.llm_engine.generate(expansion_prompt)
        if isinstance(response, dict):
            response = response.get('text')
        else:
            response = str(response)
        return query + ". " + response
    
    def retrieve_tools(self, query: str) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant tools for a given query.
        
        Args:
            query: User query string
            
        Returns:
            List of tuples (tool_name, similarity_score) sorted by relevance
        """
        if not self.tool_names or self.tool_embeddings is None:
            return []
        
        query = self.expand_query(query)
        print(f"Expanded query: {query}")
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Check if embedding generation failed
        if query_embedding is None:
            return []
            
        if self.faiss_index is not None:
            # Use FAISS for fast similarity search
            k = min(self.top_k, len(self.tool_names))
            scores, indices = self.faiss_index.search(query_embedding, k)
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.tool_names):
                    results.append((self.tool_names[idx], float(score)))
            
        else:
            # Fallback to simple cosine similarity
            similarities = np.dot(self.tool_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:self.top_k]
            
            results = []
            for idx in top_indices:
                results.append((self.tool_names[idx], float(similarities[idx])))
            
        return results
    
    def get_tool_names(self, query: str) -> List[str]:
        """
        Get just the tool names for a query (simplified interface).
        
        Args:
            query: User query stringf
            
        Returns:
            List of relevant tool names
        """
        results = self.retrieve_tools(query)
        return [tool_name for tool_name, score in results]
    
    def set_toolbox_metadata(self, metadata: Dict[str, Any]):
        """
        Set toolbox metadata for retrieved tools.
        
        Args:
            metadata: Dictionary mapping tool names to their metadata
        """
        self.toolbox_metadata = metadata
    
    def get_retrieved_tools_metadata(self, query: str) -> str:
        """
        Get metadata for retrieved tools formatted as a string.
        
        Args:
            query: User query string
            
        Returns:
            Formatted string with tool names and metadata
        """
        tool_names = self.get_tool_names(query)
        
        if not tool_names:
            return "No relevant tools found."
        
        metadata_lines = []
        for tool_name in tool_names:
            tool_metadata = self.toolbox_metadata.get(tool_name, {})
            metadata_lines.append(f"- {tool_name}: {tool_metadata}")
        
        return "\n".join(metadata_lines)
