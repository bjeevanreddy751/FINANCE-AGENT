"""
ChromaDB Vector Store Module
Manages the vector database for financial documents and transactions.
"""

from typing import List, Dict, Optional, Any
import os
import shutil

import chromadb
from chromadb.config import Settings

from config import config
from rag.embeddings import get_embeddings, get_embedding_service


class FinanceVectorStore:
    """
    ChromaDB-based vector store for financial documents and transactions.

    Stores:
    - Transaction descriptions with metadata
    - Financial summaries and insights
    - User queries and relevant context
    """

    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or config.CHROMA_DB_PATH
        self.collection_name = config.CHROMA_COLLECTION_NAME
        self.client = None
        self.collection = None
        self._initialized = False

    def initialize(self, reset: bool = False):
        """
        Initialize the ChromaDB client and collection.

        Args:
            reset: If True, delete existing database and start fresh
        """
        try:
            # Handle reset
            if reset and os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)

            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings()
            )

            # Get or create collection (get_or_create avoids the
            # "Collection already exists" error when the app is restarted).
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "FinanceIQ transaction and document embeddings"}
            )

            self._initialized = True
            return True

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            # Fallback: create an in-memory store simulation
            self._memory_store = []
            self._initialized = True
            return False

    def add_transactions(self, transactions: List[Dict], batch_size: int = 50) -> int:
        """
        Add transactions to the vector store.

        Args:
            transactions: List of transaction dictionaries
            batch_size: Number of transactions to add per batch

        Returns:
            Number of transactions added
        """
        if not self._initialized:
            self.initialize()

        if self.collection is None:
            print("Vector store is unavailable; skipping indexing.")
            return 0

        if not transactions:
            return 0

        added_count = 0

        # Process in batches
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]

            ids = []
            texts = []
            metadatas = []

            for j, txn in enumerate(batch):
                txn_id = txn.get('id') or f"txn_{i + j}_{hash(txn.get('description', ''))}"
                ids.append(txn_id)

                # Create text representation
                text = f"{txn.get('date', '')} | {txn.get('description', '')} | " \
                       f"Amount: ₹{txn.get('amount', 0):,.2f} | " \
                       f"Type: {'Credit' if txn.get('credit', 0) > txn.get('debit', 0) else 'Debit'}"

                texts.append(text)

                # Metadata
                metadata = {
                    'date': txn.get('date'),
                    'description': txn.get('description'),
                    'amount': float(txn.get('amount', 0)),
                    'type': 'credit' if txn.get('credit', 0) > txn.get('debit', 0) else 'debit',
                    'debit': float(txn.get('debit', 0)),
                    'credit': float(txn.get('credit', 0)),
                }

                # Add category if available
                if 'category' in txn and txn['category']:
                    metadata['category'] = txn['category']

                metadatas.append(metadata)

            try:
                # Generate embeddings
                embeddings = get_embeddings(texts)

                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas
                )

                added_count += len(batch)

            except Exception as e:
                print(f"Error adding batch to vector store: {e}")
                # Store in memory fallback
                if hasattr(self, '_memory_store'):
                    for j, txn in enumerate(batch):
                        self._memory_store.append({
                            'id': ids[j],
                            'text': texts[j],
                            'metadata': metadatas[j],
                            'embedding': embeddings[j].tolist() if 'embeddings' in dir() else None
                        })

        return added_count

    def add_documents(self, documents: List[Dict], collection_name: str = None) -> int:
        """
        Add arbitrary documents to the vector store.

        Args:
            documents: List of doc dicts with 'content' and 'metadata'
            collection_name: Optional collection name (uses default if None)

        Returns:
            Number of documents added
        """
        if not self._initialized:
            self.initialize()

        if not documents:
            return 0

        target_collection = self.collection
        if collection_name:
            try:
                target_collection = self.client.get_collection(name=collection_name)
            except Exception:
                target_collection = self.client.create_collection(name=collection_name)

        ids = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            doc_id = doc.get('id') or f"doc_{i}_{hash(doc.get('content', ''))}"
            ids.append(doc_id)

            texts.append(doc.get('content', ''))

            metadata = doc.get('metadata', {}).copy()
            metadata['source'] = doc.get('source', 'unknown')
            metadata['type'] = doc.get('type', 'document')
            metadatas.append(metadata)

        try:
            embeddings = get_embeddings(texts)

            target_collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )

            return len(documents)

        except Exception as e:
            print(f"Error adding documents: {e}")
            return 0

    def query(self, query_text: str, top_k: int = 5,
              filter_metadata: Dict = None) -> List[Dict]:
        """
        Query the vector store for similar documents.

        Args:
            query_text: Query text to search for
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of matching documents with similarity scores
        """
        if not self._initialized:
            self.initialize()

        if self.collection is None or not query_text:
            return []

        # Generate query embedding
        query_embedding = get_embeddings([query_text])[0]

        try:
            # Build where clause for metadata filtering
            where_clause = None
            if filter_metadata:
                where_clause = {}
                for key, value in filter_metadata.items():
                    where_clause[key] = {"$eq": value}

            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            if results and results.get('ids') and results['ids'][0]:
                ids_list = results['ids'][0]
                documents_list = results['documents'][0] if results['documents'] else []
                metadatas_list = results['metadatas'][0] if results['metadatas'] else []
                distances_list = results['distances'][0] if results['distances'] else []

                for idx in range(len(ids_list)):
                    # Convert distance to similarity score
                    distance = distances_list[idx] if idx < len(distances_list) else 1.0
                    similarity = 1.0 / (1.0 + distance)  # Convert to 0-1 similarity

                    formatted_results.append({
                        'id': ids_list[idx],
                        'content': documents_list[idx] if idx < len(documents_list) else '',
                        'metadata': metadatas_list[idx] if idx < len(metadatas_list) else {},
                        'similarity': similarity,
                        'distance': distance
                    })

            return formatted_results

        except Exception as e:
            print(f"Error querying vector store: {e}")

            # Fallback to memory store
            if hasattr(self, '_memory_store'):
                return self._memory_search(query_text, top_k, filter_metadata)

            return []

    def _memory_search(self, query_text: str, top_k: int,
                       filter_metadata: Dict = None) -> List[Dict]:
        """Fallback search using in-memory store."""
        query_embedding = get_embeddings([query_text])[0]

        results = []
        for item in self._memory_store:
            # Check metadata filter
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if item['metadata'].get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Calculate similarity
            if item.get('embedding'):
                emb = np.array(item['embedding'])
                sim = float(np.dot(query_embedding, emb) /
                           (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-10))
            else:
                # Simple string similarity fallback
                sim = self._simple_text_similarity(query_text, item['text'])

            results.append({
                'id': item['id'],
                'content': item['text'],
                'metadata': item['metadata'],
                'similarity': sim,
                'distance': 1.0 - sim
            })

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text-based similarity for fallback."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def get_all_documents(self, collection_name: str = None) -> List[Dict]:
        """
        Retrieve all documents from the collection.

        Args:
            collection_name: Optional collection name

        Returns:
            List of all documents
        """
        if not self._initialized:
            self.initialize()

        target_collection = self.collection
        if collection_name:
            try:
                target_collection = self.client.get_collection(name=collection_name)
            except Exception:
                return []

        try:
            results = target_collection.get(include=['documents', 'metadatas', 'ids'])

            documents = []
            if results and results.get('ids'):
                for i in range(len(results['ids'])):
                    doc = {
                        'id': results['ids'][i],
                        'content': results['documents'][i] if i < len(results.get('documents', [])) else '',
                        'metadata': results['metadatas'][i] if i < len(results.get('metadatas', [])) else {}
                    }
                    documents.append(doc)

            return documents

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def delete_all(self) -> bool:
        """
        Delete all documents from the vector store.

        Returns:
            True if successful
        """
        try:
            if self.collection:
                self.collection.delete()
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False

    def delete_by_metadata(self, metadata_filter: Dict) -> bool:
        """
        Delete documents matching metadata filter.

        Args:
            metadata_filter: Metadata to match for deletion

        Returns:
            True if successful
        """
        try:
            where_clause = {}
            for key, value in metadata_filter.items():
                where_clause[key] = {"$eq": value}

            self.collection.delete(where=where_clause)
            return True

        except Exception as e:
            print(f"Error deleting by metadata: {e}")
            return False

    def get_count(self) -> int:
        """Get the number of documents in the collection."""
        if not self._initialized:
            self.initialize()

        try:
            return self.collection.count()
        except Exception:
            return len(getattr(self, '_memory_store', []))

    def reset(self):
        """Reset the entire vector store."""
        self.delete_all()
        self.initialize(reset=True)


# Singleton instance
_vector_store = None

def get_vector_store(reset: bool = False) -> FinanceVectorStore:
    """
    Get the singleton vector store instance.

    Args:
        reset: If True, reset the database

    Returns:
        FinanceVectorStore instance
    """
    global _vector_store
    if _vector_store is None or reset:
        _vector_store = FinanceVectorStore()
        _vector_store.initialize(reset=reset)
    return _vector_store


# Import numpy for fallback search
import numpy as np
