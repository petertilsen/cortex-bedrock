import threading
from typing import List, Dict, Optional
import logging
import os
from cortex.constants import (
    DEFAULT_EMBEDDING_MODEL, is_bedrock_model, is_local_model, 
    get_embedding_dimension
)

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Thread-safe singleton manager supporting both Bedrock and local embedding models.
    
    Automatically detects model type and uses appropriate backend:
    - Bedrock models: Uses AWS Bedrock API
    - Local models: Uses SentenceTransformers
    """
    
    _instances: Dict[str, 'EmbeddingManager'] = {}
    _lock = threading.Lock()

    def __new__(cls, model_name: str = DEFAULT_EMBEDDING_MODEL):
        with cls._lock:
            if model_name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[model_name] = instance
                instance._initialized = False
            return cls._instances[model_name]

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Initialize embedding manager for any supported model."""
        if self._initialized:
            return
            
        self.model_name = model_name
        self._initialized = True
        self.is_bedrock = is_bedrock_model(model_name)
        self.is_local = is_local_model(model_name)
        
        if self.is_bedrock:
            self._init_bedrock()
        elif self.is_local:
            self._init_local()
        else:
            # Try Bedrock as fallback for unknown models
            logger.warning(f"Unknown model {model_name}, trying Bedrock backend")
            self.is_bedrock = True
            self._init_bedrock()

    def _init_bedrock(self, region: str = "us-east-1"):
        """Initialize Bedrock backend"""
        try:
            import boto3
            self.region = region
            self.client = boto3.client('bedrock-runtime', region_name=region)
            logger.info(f"Created EmbeddingManager for Bedrock model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock backend: {e}")
            raise

    def _init_local(self):
        """Initialize local SentenceTransformers backend"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Created EmbeddingManager for local model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize local model {self.model_name}: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """ the dimension of embeddings for this model."""
        if self.is_local and hasattr(self, 'model'):
            return self.model.get_sentence_embedding_dimension()
        else:
            return get_embedding_dimension(self.model_name)

    def get_embedding(self, content: str) -> List[float]:
        """ embeddings for text content using appropriate backend."""
        if not content or not content.strip():
            # Return zero embedding for empty content
            return [0.0] * self.get_embedding_dimension()
        
        try:
            if self.is_bedrock:
                return self._get_bedrock_embedding(content.strip())
            else:
                return self._get_local_embedding(content.strip())
        except Exception as e:
            logger.error(f"Error generating embedding with {self.model_name}: {e}")
            # Return zero embedding as fallback
            return [0.0] * self.get_embedding_dimension()

    def _get_bedrock_embedding(self, content: str) -> List[float]:
        """Get Bedrock embedding"""
        import json
        
        body = {
            "inputText": content
        }
        
        response = self.client.invoke_model(
            modelId=self.model_name,
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['embedding']

    def _get_local_embedding(self, content: str) -> List[float]:
        """ get local SentenceTransformers embedding"""
        embedding = self.model.encode(content)
        return embedding.tolist()

    def get_embeddings(self, contents: List[str]) -> List[List[float]]:
        """ embeddings for multiple text contents efficiently."""
        if not contents:
            return []
        
        processed_contents = []
        empty_indices = []
        
        for i, content in enumerate(contents):
            if not content or not content.strip():
                empty_indices.append(i)
                processed_contents.append("")  # Placeholder
            else:
                processed_contents.append(content.strip())
        
        try:
            if self.is_bedrock:
                return self._get_bedrock_embeddings(processed_contents, empty_indices)
            else:
                return self._get_local_embeddings(processed_contents, empty_indices)
        except Exception as e:
            logger.error(f"Error generating batch embeddings with {self.model_name}: {e}")
            # Return zero embeddings as fallback
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in contents]

    def _get_bedrock_embeddings(self, processed_contents: List[str], empty_indices: List[int]) -> List[List[float]]:
        """Get batch Bedrock embeddings"""
        import json
        
        # Filter out empty contents for API call
        non_empty_contents = [c for c in processed_contents if c]
        
        if not non_empty_contents:
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in processed_contents]
        
        # Get embeddings one by one (Bedrock doesn't support batch)
        batch_embeddings = []
        for content in non_empty_contents:
            body = {"inputText": content}
            response = self.client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(body)
            )
            response_body = json.loads(response['body'].read())
            batch_embeddings.append(response_body['embedding'])
        
        # Reconstruct full list with zero embeddings for empty contents
        embeddings = []
        non_empty_idx = 0
        dim = self.get_embedding_dimension()
        
        for i, content in enumerate(processed_contents):
            if i in empty_indices:
                embeddings.append([0.0] * dim)
            else:
                embeddings.append(batch_embeddings[non_empty_idx])
                non_empty_idx += 1
                
        return embeddings

    def _get_local_embeddings(self, processed_contents: List[str], empty_indices: List[int]) -> List[List[float]]:
        """ get batch local embeddings"""
        # Filter out empty contents
        non_empty_contents = [c for c in processed_contents if c]
        
        if not non_empty_contents:
            # All contents are empty
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in processed_contents]
        
        # Use SentenceTransformers batch encoding
        batch_embeddings = self.model.encode(non_empty_contents)
        
        embeddings = []
        non_empty_idx = 0
        dim = self.get_embedding_dimension()
        
        for i, content in enumerate(processed_contents):
            if i in empty_indices:
                embeddings.append([0.0] * dim)
            else:
                embeddings.append(batch_embeddings[non_empty_idx].tolist())
                non_empty_idx += 1
                
        return embeddings

    @classmethod
    def get_manager(cls, model_name: str = DEFAULT_EMBEDDING_MODEL) -> 'EmbeddingManager':
        """get or create an embedding manager for the specified model."""
        return cls(model_name)

    @classmethod
    def clear_cache(cls):
        with cls._lock:
            cls._instances.clear()
        logger.info("Cleared embedding manager cache")
