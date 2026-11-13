"""
Agent Orchestrator
Coordinates all FinanceIQ agents in a unified workflow.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import json

from agents.parser_agent import ParserAgent, parse_statement
from agents.categorizer_agent import CategorizerAgent
from agents.analysis_agent import AnalysisAgent, AnalysisResult
from agents.advisor_agent import AdvisorAgent, AdvisorResponse
from rag.rag_pipeline import FinancialRAGPipeline, get_rag_pipeline
from rag.vector_store import get_vector_store, FinanceVectorStore


@dataclass
class ProcessingState:
    """Tracks the state of statement processing."""
    raw_transactions: List[Dict] = field(default_factory=list)
    categorized_transactions: List[Dict] = field(default_factory=list)
    analysis: Optional[AnalysisResult] = None
    advisor_response: Optional[AdvisorResponse] = None
    sources: List[Dict] = field(default_factory=list)


class AgentOrchestrator:
    """
    Orchestrates the multi-agent workflow.

    Workflow:
    1. Parser Agent - Extract transactions from statements
    2. Categorizer Agent - Classify transactions into categories
    3. Analysis Agent - Calculate financial metrics
    4. Advisor Agent - Generate personalized insights
    5. RAG Pipeline - Enable semantic search and Q&A
    """

    def __init__(self, llm=None):
        # Initialize agents
        self.parser = ParserAgent()
        self.categorizer = CategorizerAgent()
        self.analysis = AnalysisAgent()
        self.advisor = AdvisorAgent(llm)

        # Set LLM for advisor and RAG
        if llm:
            self.advisor.set_llm(llm)
            rag_pipeline = get_rag_pipeline()
            rag_pipeline.set_llm(llm)

        # Processing state
        self.state = ProcessingState()

        # RAG pipeline
        self.rag_pipeline = get_rag_pipeline()
        self.vector_store = get_vector_store()

        # Stats
        self.processing_history = []

    def process_statement(self, file_bytes: bytes, file_type: str,
                          filename: str = "statement") -> Dict:
        """
        Process a complete bank statement through all agents.

        Args:
            file_bytes: File content as bytes
            file_type: 'pdf' or 'csv'
            filename: Name of the file

        Returns:
            Processing result with all extracted data
        """
        result = {
            'success': False,
            'step': 'start',
            'message': '',
            'transactions': [],
            'analysis': None,
            'insights': None,
            'error': None
        }

        # Step 1: Parse the statement
        result['step'] = 'parsing'
        parse_result = self.parser.parse_file(file_bytes, file_type, filename)

        if not parse_result['success']:
            result['error'] = parse_result.get('error', 'Unknown parsing error')
            result['message'] = f"Failed to parse {file_type.upper()} file: {result['error']}"
            return result

        result['message'] = f"Parsed {parse_result['stats'].get('valid', 0)} transactions from {filename}"
        result['stats'] = parse_result['stats']

        # Step 2: Merge transactions
        new_transactions = parse_result['transactions']
        added_count = self.parser.merge_transactions(new_transactions)
        self.state.raw_transactions = self.parser.transactions.copy()

        result['step'] = 'categorizing'
        result['message'] = f"Merged {added_count} new transactions"

        # Step 3: Categorize transactions
        categorized = self.categorizer.apply_rule_based_categories(
            self.state.raw_transactions
        )
        self.state.categorized_transactions = categorized

        # Update parser transactions with categories
        for i, txn in enumerate(self.parser.transactions):
            if i < len(categorized):
                txn['category'] = categorized[i].get('category', 'Uncategorized')
                txn['category_source'] = categorized[i].get('category_source', 'rule-based')

        result['step'] = 'analyzing'
        result['message'] = "Categorized transactions"

        # Step 4: Analyze
        df = self.parser.get_transactions_df()
        if not df.empty:
            # Add categories to DataFrame
            if 'category' not in df.columns:
                df['category'] = 'Uncategorized'

            # Add categorized data
            for category in ['category', 'category_source', 'category_confidence']:
                if category in df.columns:
                    pass  # Keep existing columns
                else:
                    df[category] = None

            # Apply categorized data
            for i, cat_txn in enumerate(categorized):
                if i < len(df):
                    df.at[i, 'category'] = cat_txn.get('category', 'Uncategorized')
                    df.at[i, 'category_source'] = cat_txn.get('category_source', 'rule-based')

            # Perform analysis
            analysis_result = self.analysis.analyze(df)
            self.state.analysis = analysis_result
            result['analysis'] = self._serialize_analysis(analysis_result)
            result['message'] = "Analysis complete"

            # Step 5: Generate insights
            result['step'] = 'advising'
            advisor_response = self.advisor.get_insights(analysis_result)
            self.state.advisor_response = advisor_response
            result['insights'] = advisor_response.answer
            result['suggestions'] = advisor_response.suggestions
            result['warnings'] = advisor_response.warnings
            result['message'] = "Insights generated"

            # Step 6: Add to RAG
            result['step'] = 'indexing'
            self._index_transactions_for_rag(df)
            result['message'] = "Indexed for search"

        result['success'] = True
        result['transactions'] = self._serialize_transactions(df)

        # Record history
        self.processing_history.append({
            'filename': filename,
            'file_type': file_type,
            'timestamp': pd.Timestamp.now().isoformat(),
            'transaction_count': len(df),
            'success': True
        })

        return result

    def process_sample_data(self, sample_csv_path: str = None) -> Dict:
        """
        Process sample data for demonstration.

        Args:
            sample_csv_path: Path to sample CSV file

        Returns:
            Processing result
        """
        import os

        if sample_csv_path is None:
            # Default path
            sample_csv_path = os.path.join(
                os.path.dirname(__file__),
                '..', 'data', 'sample_transactions.csv'
            )

        if not os.path.exists(sample_csv_path):
            return {
                'success': False,
                'error': f"Sample file not found: {sample_csv_path}"
            }

        with open(sample_csv_path, 'rb') as f:
            content = f.read()

        return self.process_statement(content, 'csv', 'sample_transactions.csv')

    def query(self, question: str) -> Dict:
        """
        Answer a question using the RAG pipeline and available data.

        Args:
            question: User's question

        Returns:
            Answer dictionary
        """
        # Get analysis results
        analysis_data = None
        if self.state.analysis:
            analysis_data = {
                'total_income': self.state.analysis.total_income,
                'total_expenses': self.state.analysis.total_expenses,
                'net_savings': self.state.analysis.net_savings,
                'savings_rate': self.state.analysis.savings_rate,
                'highest_spending_category': self.state.analysis.highest_spending_category,
                'highest_category_amount': self.state.analysis.highest_category_amount,
                'average_daily_expenditure': self.state.analysis.average_daily_expenditure,
                'monthly_breakdown': self.state.analysis.monthly_breakdown,
                'category_breakdown': self.state.analysis.category_breakdown,
                'unusual_transactions': self.state.analysis.unusual_transactions,
            }

        # Get transaction data for context
        transaction_data = None
        if self.state.categorized_transactions:
            transaction_data = {
                'summary': {
                    'total_transactions': len(self.state.categorized_transactions),
                    'date_range': self.state.analysis.date_range if self.state.analysis else 'N/A',
                },
                'recent_transactions': self.state.categorized_transactions[-10:],
            }

        # Query the RAG pipeline
        result = self.rag_pipeline.query(
            user_query=question,
            transaction_data=transaction_data,
            analysis_results=analysis_data,
            use_lightweight=True  # Use keyword-based intent for reliability
        )

        return {
            'question': question,
            'answer': result['answer'],
            'intent': result['intent'],
            'sources': result['sources'],
            'needs_llm': result.get('needs_llm', False)
        }

    def get_summary(self) -> Dict:
        """Get a summary of the current processing state."""
        summary = {
            'total_transactions': len(self.state.raw_transactions),
            'categorized_transactions': len(self.state.categorized_transactions),
            'analysis_complete': self.state.analysis is not None,
        }

        if self.state.analysis:
            summary.update({
                'total_income': self.state.analysis.total_income,
                'total_expenses': self.state.analysis.total_expenses,
                'net_savings': self.state.analysis.net_savings,
                'savings_rate': self.state.analysis.savings_rate,
                'date_range': self.state.analysis.date_range,
            })

        return summary

    def get_monthly_data(self) -> pd.DataFrame:
        """Get monthly income vs expenses data for charts."""
        return self.analysis.get_monthly_data()

    def get_category_chart_data(self) -> Tuple[List[str], List[float]]:
        """Get (categories, amounts) for the category charts."""
        return self.analysis.get_category_chart_data()

    def get_transactions_df(self) -> pd.DataFrame:
        """Get the current transactions as DataFrame."""
        return self.parser.get_transactions_df()

    def get_analysis(self) -> Optional[AnalysisResult]:
        """Get the last analysis result."""
        return self.state.analysis

    def get_raw_transactions(self) -> List[Dict]:
        """Get raw (uncategorized) transactions."""
        return self.state.raw_transactions.copy()

    def get_categorized_transactions(self) -> List[Dict]:
        """Get categorized transactions."""
        return self.state.categorized_transactions.copy()

    def update_category(self, transaction_index: int, new_category: str) -> bool:
        """
        Update the category of a specific transaction.

        Args:
            transaction_index: Index of the transaction
            new_category: New category name

        Returns:
            True if successful
        """
        if transaction_index < 0 or transaction_index >= len(self.state.categorized_transactions):
            return False

        # Update categorized transaction
        self.state.categorized_transactions[transaction_index]['category'] = new_category
        self.state.categorized_transactions[transaction_index]['category_source'] = 'user-corrected'

        # Update raw transactions too
        if transaction_index < len(self.state.raw_transactions):
            self.state.raw_transactions[transaction_index]['category'] = new_category
            self.state.raw_transactions[transaction_index]['category_source'] = 'user-corrected'

        # Update DataFrame if exists
        df = self.parser.get_transactions_df()
        if transaction_index < len(df):
            df.at[transaction_index, 'category'] = new_category
            df.at[transaction_index, 'category_source'] = 'user-corrected'

        # Re-analyze with updated categories
        if not df.empty:
            self.state.analysis = self.analysis.analyze(df)

        return True

    def reset(self):
        """Reset the orchestrator state."""
        self.parser.clear_transactions()
        self.categorizer.clear_user_corrections()
        self.analysis.clear_cache()
        self.state = ProcessingState()

        # Reset vector store
        self.vector_store.reset()

        self.processing_history = []

    def _serialize_analysis(self, analysis: AnalysisResult) -> Dict:
        """Serialize analysis for JSON output."""
        return {
            'total_income': analysis.total_income,
            'total_expenses': analysis.total_expenses,
            'net_savings': analysis.net_savings,
            'savings_rate': analysis.savings_rate,
            'transaction_count': analysis.transaction_count,
            'date_range': analysis.date_range,
            'monthly_breakdown': analysis.monthly_breakdown,
            'category_breakdown': analysis.category_breakdown,
            'highest_spending_category': analysis.highest_spending_category,
            'highest_category_amount': analysis.highest_category_amount,
            'average_daily_expenditure': analysis.average_daily_expenditure,
            'month_over_month_changes': analysis.month_over_month_changes,
            'unusual_transactions': analysis.unusual_transactions,
        }

    def _serialize_transactions(self, df: pd.DataFrame) -> List[Dict]:
        """Serialize DataFrame to list of dicts."""
        if df.empty:
            return []

        result = []
        for _, row in df.iterrows():
            txn = {
                'date': row.get('date'),
                'description': str(row.get('description', '')),
                'amount': float(row.get('amount', 0)),
                'debit': float(row.get('debit', 0)),
                'credit': float(row.get('credit', 0)),
            }

            if 'category' in df.columns:
                txn['category'] = str(row.get('category', 'Uncategorized'))

            if 'balance' in df.columns and pd.notna(row.get('balance')):
                txn['balance'] = float(row['balance'])

            result.append(txn)

        return result

    def _index_transactions_for_rag(self, df: pd.DataFrame):
        """Index transactions in the vector store."""
        transactions = []

        for _, row in df.iterrows():
            txn = {
                'date': row.get('date'),
                'description': str(row.get('description', '')),
                'amount': float(row.get('amount', 0)),
                'debit': float(row.get('debit', 0)),
                'credit': float(row.get('credit', 0)),
            }

            if 'category' in df.columns and pd.notna(row.get('category')):
                txn['category'] = str(row['category'])

            if 'balance' in df.columns and pd.notna(row.get('balance')):
                txn['balance'] = float(row['balance'])

            transactions.append(txn)

        self.rag_pipeline.add_transactions_to_rag(transactions)


# Singleton orchestrator
_orchestrator = None

def get_orchestrator(llm=None) -> AgentOrchestrator:
    """Get the singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(llm)
    return _orchestrator

def reset_orchestrator():
    """Reset the orchestrator."""
    global _orchestrator
    if _orchestrator:
        _orchestrator.reset()
    _orchestrator = None
