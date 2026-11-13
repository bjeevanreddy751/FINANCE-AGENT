"""
RAG Retriever Module
Retrieves relevant financial context for user queries.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from rag.vector_store import get_vector_store, FinanceVectorStore
from rag.embeddings import get_embedding_service, similarity_score
import numpy as np


@dataclass
class RetrievedContext:
    """A single retrieved context item."""
    content: str
    metadata: Dict
    similarity_score: float
    source_type: str  # 'transaction', 'document', 'summary'


@dataclass
class RetrievalResult:
    """Complete retrieval result."""
    contexts: List[RetrievedContext]
    query: str
    intent: Optional[str] = None
    total_results: int = 0


class FinancialRetriever:
    """
    Retrieves relevant financial context for user queries using RAG.

    Capabilities:
    - Semantic search through transaction descriptions
    - Category-based filtering
    - Date range filtering
    - Hybrid search (keyword + semantic)
    """

    def __init__(self, vector_store: FinanceVectorStore = None):
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = get_embedding_service()

    def retrieve_for_query(self, query: str, top_k: int = 5,
                           filter_by: Dict = None) -> RetrievalResult:
        """
        Retrieve relevant context for a user query.

        Args:
            query: User's question/query
            top_k: Maximum number of results
            filter_by: Optional metadata filters

        Returns:
            RetrievalResult with relevant contexts
        """
        # Query the vector store
        raw_results = self.vector_store.query(
            query_text=query,
            top_k=top_k,
            filter_metadata=filter_by
        )

        # Convert to RetrievedContext objects
        contexts = []
        for result in raw_results:
            source_type = result['metadata'].get('type', 'transaction')
            if 'category' in result['metadata']:
                source_type = 'transaction'

            contexts.append(RetrievedContext(
                content=result['content'],
                metadata=result['metadata'],
                similarity_score=result['similarity'],
                source_type=source_type
            ))

        return RetrievalResult(
            contexts=contexts,
            query=query,
            total_results=len(contexts)
        )

    def retrieve_transactions(self, query: str, top_k: int = 10,
                              category: str = None) -> RetrievalResult:
        """
        Retrieve transactions matching a query.

        Args:
            query: Search query (semantic or keyword)
            top_k: Number of transactions to return
            category: Optional category filter

        Returns:
            RetrievalResult with transaction contexts
        """
        filter_by = None
        if category:
            filter_by = {'category': category}

        return self.retrieve_for_query(query, top_k=top_k, filter_by=filter_by)

    def retrieve_by_date_range(self, start_date: str, end_date: str,
                                top_k: int = 50) -> List[RetrievedContext]:
        """
        Retrieve transactions within a date range.

        Since ChromaDB filters are limited, this retrieves all and filters in memory.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            top_k: Maximum transactions to return

        Returns:
            List of RetrievedContext objects
        """
        # Get all documents
        all_docs = self.vector_store.get_all_documents()

        # Filter by date range in memory
        filtered = []
        for doc in all_docs:
            date_str = doc['metadata'].get('date', '')
            if start_date <= date_str <= end_date:
                filtered.append(RetrievedContext(
                    content=doc['content'],
                    metadata=doc['metadata'],
                    similarity_score=1.0,  # No similarity scoring for date-based
                    source_type='transaction'
                ))

        # Sort by date
        filtered.sort(
            key=lambda x: x.metadata.get('date', ''),
            reverse=True
        )

        return filtered[:top_k]

    def retrieve_category_summary(self, category: str) -> RetrievalResult:
        """
        Retrieve all transactions for a specific category.

        Args:
            category: Category name

        Returns:
            RetrievalResult with category transactions
        """
        return self.retrieve_transactions(
            query=f"transactions in {category} category",
            top_k=100,
            filter_by={'category': category}
        )

    def hybrid_search(self, query: str, keywords: List[str] = None,
                      top_k: int = 10) -> RetrievalResult:
        """
        Perform hybrid search combining semantic and keyword matching.

        Args:
            query: Semantic query
            keywords: Additional keyword filters
            top_k: Number of results

        Returns:
            Combined retrieval result
        """
        # Get semantic results
        semantic_result = self.retrieve_for_query(query, top_k=top_k)

        # If keywords provided, filter results
        if keywords:
            filtered_contexts = []
            for context in semantic_result.contexts:
                content_lower = context.content.lower()
                if any(kw.lower() in content_lower for kw in keywords):
                    filtered_contexts.append(context)

            if filtered_contexts:
                semantic_result.contexts = filtered_contexts
                semantic_result.total_results = len(filtered_contexts)

        return semantic_result

    def retrieve_relevant_context(self, query: str,
                                   transaction_data: Dict = None,
                                   top_k: int = 5) -> str:
        """
        Retrieve and format relevant context for LLM prompt.

        Args:
            query: User's query
            transaction_data: Available transaction data summary
            top_k: Number of context items

        Returns:
            Formatted context string for LLM
        """
        retrieval = self.retrieve_for_query(query, top_k=top_k)

        if not retrieval.contexts:
            # No vector store results, provide data summary
            if transaction_data:
                return self._format_data_summary(transaction_data)
            return "No relevant transaction data available in the system."

        # Format retrieved contexts
        context_parts = ["## Relevant Transaction Context\n"]

        for i, ctx in enumerate(retrieval.contexts, 1):
            context_parts.append(f"\n### Match {i} (Relevance: {ctx.similarity_score:.2f})")
            context_parts.append(f"**Date:** {ctx.metadata.get('date', 'N/A')}")
            context_parts.append(f"**Description:** {ctx.metadata.get('description', ctx.content)}")
            context_parts.append(f"**Amount:** ₹{ctx.metadata.get('amount', 0):,.2f}")
            context_parts.append(f"**Type:** {ctx.metadata.get('type', 'N/A')}")

            if ctx.metadata.get('category'):
                context_parts.append(f"**Category:** {ctx.metadata['category']}")

            context_parts.append(f"**Details:** {ctx.content}")

        return "\n".join(context_parts)

    def _format_data_summary(self, data: Dict) -> str:
        """Format transaction data summary as context."""
        parts = []

        if data.get('summary'):
            parts.append("## Financial Summary")
            summary = data['summary']
            parts.append(f"- Total Income: ₹{summary.get('total_income', 0):,.2f}")
            parts.append(f"- Total Expenses: ₹{summary.get('total_expenses', 0):,.2f}")
            parts.append(f"- Net Savings: ₹{summary.get('net_savings', 0):,.2f}")
            parts.append("")

        if data.get('monthly_breakdown'):
            parts.append("## Monthly Breakdown")
            for month, stats in data['monthly_breakdown'].items():
                parts.append(f"\n### {month}")
                parts.append(f"- Income: ₹{stats.get('income', 0):,.2f}")
                parts.append(f"- Expenses: ₹{stats.get('expenses', 0):,.2f}")
                parts.append(f"- Savings: ₹{stats.get('savings', 0):,.2f}")
            parts.append("")

        if data.get('category_breakdown'):
            parts.append("## Category Breakdown")
            for category, stats in list(data['category_breakdown'].items())[:5]:
                parts.append(f"- {category}: ₹{stats.get('total', 0):,.2f}")
            parts.append("")

        return "\n".join(parts)

    def clear_cache(self):
        """Clear any internal caching."""
        pass  # No caching in this implementation


class ContextBuilder:
    """
    Builds comprehensive context for RAG queries.
    Combines retrieved context with transaction data summaries.
    """

    def __init__(self, retriever: FinancialRetriever = None):
        self.retriever = retriever or FinancialRetriever()

    def build_context(self, query: str,
                      transaction_df=None,
                      analysis_results: Dict = None,
                      include_raw_data: bool = False) -> Dict:
        """
        Build comprehensive context for a query.

        Args:
            query: User's query
            transaction_df: Optional transaction DataFrame
            analysis_results: Optional analysis results
            include_raw_data: Whether to include raw transaction data

        Returns:
            Dictionary with context components
        """
        context = {
            'query': query,
            'retrieved_context': '',
            'data_summary': '',
            'raw_transactions': None
        }

        # Get retrieved context from RAG
        if self.retriever.vector_store.get_count() > 0:
            context['retrieved_context'] = self.retriever.retrieve_relevant_context(
                query, top_k=3
            )

        # Add data summary if available
        if analysis_results:
            context['data_summary'] = self._format_analysis_summary(analysis_results)

        # Include raw data if requested and available
        if include_raw_data and transaction_df is not None:
            context['raw_transactions'] = self._format_recent_transactions(transaction_df)

        return context

    def _format_analysis_summary(self, analysis: Dict) -> str:
        """Format analysis results as context string."""
        parts = ["## Financial Analysis Summary\n"]

        if 'total_income' in analysis:
            parts.append(f"- **Total Income:** ₹{analysis['total_income']:,.2f}")

        if 'total_expenses' in analysis:
            parts.append(f"- **Total Expenses:** ₹{analysis['total_expenses']:,.2f}")

        if 'net_savings' in analysis:
            parts.append(f"- **Net Savings:** ₹{analysis['net_savings']:,.2f}")

        if 'savings_rate' in analysis:
            parts.append(f"- **Savings Rate:** {analysis['savings_rate']:.1f}%")

        if analysis.get('highest_spending_category'):
            parts.append(
                f"- **Highest Spending Category:** "
                f"{analysis['highest_spending_category']} "
                f"(₹{analysis.get('highest_category_amount', 0):,.2f})"
            )

        if analysis.get('monthly_changes'):
            parts.append("\n### Month-over-Month Changes")
            for month, change in analysis['monthly_changes'].items():
                direction = "increase" if change['percent_change'] > 0 else "decrease"
                parts.append(
                    f"- {month}: {direction} of "
                    f"₹{abs(change['absolute_change']):,.2f} "
                    f"({change['percent_change']:+.1f}%)"
                )

        if analysis.get('unusual_transactions'):
            parts.append("\n### Unusual Transactions")
            for txn in analysis['unusual_transactions'][:3]:
                parts.append(
                    f"- **{txn['date']}**: {txn['description']} - "
                    f"₹{txn['amount']:,.2f} ({txn['reason']})"
                )

        return "\n".join(parts)

    def _format_recent_transactions(self, df) -> str:
        """Format recent transactions as context string."""
        if df is None or df.empty:
            return ""

        parts = ["## Recent Transactions\n"]

        recent = df.head(15)
        for _, row in recent.iterrows():
            date = row.get('date', 'N/A')
            desc = row.get('description', 'N/A')
            amount = row.get('amount', 0)
            category = row.get('category', 'Uncategorized')

            parts.append(f"- **{date}**: {desc} - ₹{amount:,.2f} [{category}]")

        return "\n".join(parts)


# Singleton instances
_retriever = None
_context_builder = None


def get_retriever() -> FinancialRetriever:
    """Get the singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = FinancialRetriever()
    return _retriever


def get_context_builder() -> ContextBuilder:
    """Get the singleton context builder instance."""
    global _context_builder
    if _context_builder is None:
        _context_builder = ContextBuilder()
    return _context_builder
