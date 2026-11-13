"""
Financial Analysis Agent
Performs comprehensive financial calculations and analysis.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np

from utils.calculations import (
    calculate_total_income,
    calculate_total_expenses,
    calculate_net_savings,
    calculate_monthly_breakdown,
    calculate_category_breakdown,
    calculate_highest_spending_category,
    calculate_average_daily_expenditure,
    calculate_monthly_savings_rate,
    calculate_month_over_month_change,
    detect_unusual_transactions,
    get_income_vs_expenses_by_month,
    filter_transactions,
    get_recent_transactions,
    calculate_running_balance
)


@dataclass
class AnalysisResult:
    """Comprehensive financial analysis result."""
    total_income: float
    total_expenses: float
    net_savings: float
    savings_rate: float
    transaction_count: int
    date_range: str
    monthly_breakdown: Dict
    category_breakdown: Dict
    highest_spending_category: Optional[str]
    highest_category_amount: float
    average_daily_expenditure: float
    month_over_month_changes: Dict
    unusual_transactions: List[Dict]
    recent_transactions: List[Dict]


class AnalysisAgent:
    """
    Agent for performing financial analysis on transaction data.

    Uses Pandas for accurate numerical calculations.
    NEVER allows LLM to invent financial totals.
    """

    def __init__(self):
        self.transactions_df: pd.DataFrame = None
        self.analysis_cache: Dict = {}
        self.last_analysis: AnalysisResult = None

    def analyze(self, transactions_df: pd.DataFrame) -> AnalysisResult:
        """
        Perform comprehensive financial analysis.

        Args:
            transactions_df: DataFrame with transactions

        Returns:
            AnalysisResult with all calculated metrics
        """
        if transactions_df.empty:
            return self._empty_result()

        # Store for later use
        self.transactions_df = transactions_df.copy()

        # Ensure proper types
        df = self._prepare_dataframe(transactions_df)

        # Calculate core metrics
        total_income = calculate_total_income(df)
        total_expenses = calculate_total_expenses(df)
        net_savings = calculate_net_savings(df)

        # Calculate savings rate
        savings_rate = 0.0
        if total_income > 0:
            savings_rate = (net_savings / total_income) * 100

        # Date range
        date_range = self._get_date_range(df)

        # Monthly breakdown
        monthly_breakdown = calculate_monthly_breakdown(df)

        # Category breakdown (if categories exist)
        category_breakdown = {}
        if 'category' in df.columns:
            category_breakdown = calculate_category_breakdown(df)

        # Highest spending category
        highest_spending_category = None
        highest_category_amount = 0.0
        if category_breakdown:
            highest_spending_category = calculate_highest_spending_category(df)
            if highest_spending_category and highest_spending_category in category_breakdown:
                highest_category_amount = category_breakdown[highest_spending_category]['expenses']

        # Average daily expenditure
        avg_daily = calculate_average_daily_expenditure(df)

        # Month-over-month changes
        month_over_month = calculate_month_over_month_change(df)

        # Unusual transactions
        unusual = detect_unusual_transactions(df)

        # Recent transactions
        recent = get_recent_transactions(df, limit=10)
        recent_list = self._df_to_list(recent)

        result = AnalysisResult(
            total_income=total_income,
            total_expenses=total_expenses,
            net_savings=net_savings,
            savings_rate=savings_rate,
            transaction_count=len(df),
            date_range=date_range,
            monthly_breakdown=monthly_breakdown,
            category_breakdown=category_breakdown,
            highest_spending_category=highest_spending_category,
            highest_category_amount=highest_category_amount,
            average_daily_expenditure=avg_daily,
            month_over_month_changes=month_over_month,
            unusual_transactions=unusual,
            recent_transactions=recent_list
        )

        # Cache the result
        self.last_analysis = result
        self.analysis_cache = self._serialize_result(result)

        return result

    def get_monthly_data(self) -> pd.DataFrame:
        """
        Get monthly income vs expenses data for charting.

        Returns:
            DataFrame with month, income, expenses, savings columns
        """
        if self.transactions_df is None or self.transactions_df.empty:
            return pd.DataFrame(columns=['month', 'income', 'expenses', 'savings'])

        return get_income_vs_expenses_by_month(self.transactions_df)

    def get_category_chart_data(self) -> Tuple[List[str], List[float]]:
        """
        Get category data for pie chart.

        Returns:
            Tuple of (categories, amounts)
        """
        if not self.last_analysis or not self.last_analysis.category_breakdown:
            return [], []

        categories = []
        amounts = []

        for category, data in self.last_analysis.category_breakdown.items():
            if data['expenses'] > 0:  # Only include expense categories for pie chart
                categories.append(category)
                amounts.append(data['expenses'])

        # Sort by amount descending
        sorted_pairs = sorted(zip(categories, amounts), key=lambda x: x[1], reverse=True)
        return [c for c, _ in sorted_pairs], [a for _, a in sorted_pairs]

    def analyze_category(self, category: str) -> Dict:
        """
        Get detailed analysis for a specific category.

        Args:
            category: Category name

        Returns:
            Category analysis dictionary
        """
        if self.transactions_df is None or self.transactions_df.empty:
            return {'error': 'No data available'}

        df = self.transactions_df
        if 'category' not in df.columns:
            return {'error': 'Transactions not categorized'}

        category_df = df[df['category'] == category]

        if category_df.empty:
            return {'error': f'No transactions in category: {category}'}

        return {
            'category': category,
            'total_spent': float(category_df['debit'].sum()),
            'total_income': float(category_df['credit'].sum()),
            'transaction_count': len(category_df),
            'average_transaction': float(category_df['amount'].mean()),
            'largest_transaction': float(category_df['amount'].max()),
            'smallest_transaction': float(category_df['amount'].min()),
            'transactions': self._df_to_list(category_df.head(20))
        }

    def compare_months(self, month1: str, month2: str) -> Dict:
        """
        Compare financial metrics between two months.

        Args:
            month1: First month (YYYY-MM format)
            month2: Second month (YYYY-MM format)

        Returns:
            Comparison dictionary
        """
        if self.transactions_df is None or self.transactions_df.empty:
            return {'error': 'No data available'}

        df = self.transactions_df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.to_period('M').astype(str)

        month1_df = df[df['month'] == month1]
        month2_df = df[df['month'] == month2]

        def month_stats(mdf):
            return {
                'income': float(mdf['credit'].sum()),
                'expenses': float(mdf['debit'].sum()),
                'savings': float(mdf['credit'].sum() - mdf['debit'].sum()),
                'transaction_count': len(mdf)
            }

        stats1 = month_stats(month1_df) if not month1_df.empty else {'income': 0, 'expenses': 0, 'savings': 0, 'transaction_count': 0}
        stats2 = month_stats(month2_df) if not month2_df.empty else {'income': 0, 'expenses': 0, 'savings': 0, 'transaction_count': 0}

        return {
            'month1': {
                'name': month1,
                **stats1
            },
            'month2': {
                'name': month2,
                **stats2
            },
            'changes': {
                'income_change': stats2['income'] - stats1['income'],
                'expense_change': stats2['expenses'] - stats1['expenses'],
                'savings_change': stats2['savings'] - stats1['savings'],
            }
        }

    def search_transactions(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search transactions by description.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching transactions
        """
        if self.transactions_df is None or self.transactions_df.empty:
            return []

        df = self.transactions_df
        if 'description' not in df.columns:
            return []

        query_lower = query.lower()
        mask = df['description'].str.lower().str.contains(query_lower, na=False)
        results = df[mask].head(limit)

        return self._df_to_list(results)

    def get_transactions_for_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Get transactions within a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of transactions in range
        """
        if self.transactions_df is None or self.transactions_df.empty:
            return []

        df = self.transactions_df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        filtered = df[(df['date'] >= start) & (df['date'] <= end)]

        return self._df_to_list(filtered)

    def get_analysis_summary(self) -> Dict:
        """
        Get a summary of the last analysis.

        Returns:
            Summary dictionary
        """
        if self.last_analysis is None:
            return {'error': 'No analysis performed yet'}

        return {
            'total_income': self.last_analysis.total_income,
            'total_expenses': self.last_analysis.total_expenses,
            'net_savings': self.last_analysis.net_savings,
            'savings_rate': self.last_analysis.savings_rate,
            'transaction_count': self.last_analysis.transaction_count,
            'date_range': self.last_analysis.date_range,
            'highest_spending_category': self.last_analysis.highest_spending_category,
            'highest_category_amount': self.last_analysis.highest_category_amount,
            'average_daily_expenditure': self.last_analysis.average_daily_expenditure,
            'monthly_breakdown': self.last_analysis.monthly_breakdown,
            'category_breakdown': self.last_analysis.category_breakdown,
        }

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for analysis."""
        df = df.copy()

        # Accept raw statement headers such as "Date" / "Debit" / "Credit".
        df.columns = [
            str(col).strip().lower().replace(' ', '_').replace('-', '_')
            for col in df.columns
        ]

        # Drop duplicate headers (e.g. a caller-supplied "date" plus a "Date" column).
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Ensure numeric columns
        for col in ['amount', 'debit', 'credit']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Drop rows without dates
        df = df.dropna(subset=['date'])

        return df

    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Get formatted date range."""
        if 'date' not in df.columns or df.empty:
            return "N/A"

        min_date = df['date'].min()
        max_date = df['date'].max()

        if pd.isna(min_date) or pd.isna(max_date):
            return "N/A"

        return f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"

    def _df_to_list(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to list of dictionaries."""
        result = []

        for _, row in df.iterrows():
            txn = {
                'date': row.get('date').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else None,
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

    def _empty_result(self) -> AnalysisResult:
        """Return an empty analysis result."""
        return AnalysisResult(
            total_income=0,
            total_expenses=0,
            net_savings=0,
            savings_rate=0,
            transaction_count=0,
            date_range="N/A",
            monthly_breakdown={},
            category_breakdown={},
            highest_spending_category=None,
            highest_category_amount=0,
            average_daily_expenditure=0,
            month_over_month_changes={},
            unusual_transactions=[],
            recent_transactions=[]
        )

    def _serialize_result(self, result: AnalysisResult) -> Dict:
        """Serialize AnalysisResult for caching."""
        return {
            'total_income': result.total_income,
            'total_expenses': result.total_expenses,
            'net_savings': result.net_savings,
            'savings_rate': result.savings_rate,
            'transaction_count': result.transaction_count,
            'date_range': result.date_range,
            'monthly_breakdown': result.monthly_breakdown,
            'category_breakdown': result.category_breakdown,
            'highest_spending_category': result.highest_spending_category,
            'highest_category_amount': result.highest_category_amount,
            'average_daily_expenditure': result.average_daily_expenditure,
            'month_over_month_changes': result.month_over_month_changes,
            'unusual_transactions': result.unusual_transactions,
        }

    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache = {}
        self.last_analysis = None
        self.transactions_df = None


# For LCEL integration
from langchain_core.tools import tool

@tool
def analyze_financials_tool(transactions_json: str) -> str:
    """
    Analyze financial transactions and return summary.

    Args:
        transactions_json: JSON array of transactions

    Returns:
        JSON with financial analysis summary
    """
    import json
    import pandas as pd

    transactions = json.loads(transactions_json)
    df = pd.DataFrame(transactions)

    agent = AnalysisAgent()
    analysis = agent.analyze(df)

    summary = {
        'total_income': analysis.total_income,
        'total_expenses': analysis.total_expenses,
        'net_savings': analysis.net_savings,
        'savings_rate': analysis.savings_rate,
        'transaction_count': analysis.transaction_count,
        'date_range': analysis.date_range,
        'highest_spending_category': analysis.highest_spending_category,
        'monthly_breakdown': analysis.monthly_breakdown,
    }

    return json.dumps(summary, indent=2, default=str)
