"""
Tests for FinanceIQ Analysis Agent and Calculations
"""

import pytest
import pandas as pd
import numpy as np
from agents.analysis_agent import AnalysisAgent, AnalysisResult
from utils.calculations import (
    calculate_total_income,
    calculate_total_expenses,
    calculate_net_savings,
    calculate_monthly_breakdown,
    calculate_category_breakdown,
    detect_unusual_transactions,
    calculate_month_over_month_change,
    filter_transactions,
    get_recent_transactions,
)


class TestCalculations:
    """Tests for financial calculation utilities."""

    def test_total_income_calculation(self):
        """Test total income calculation."""
        df = pd.DataFrame({
            'credit': [50000, 10000, 5000],
            'debit': [500, 200, 300],
        })

        income = calculate_total_income(df)
        assert income == 65000

    def test_total_expenses_calculation(self):
        """Test total expenses calculation."""
        df = pd.DataFrame({
            'credit': [0, 0, 0],
            'debit': [500, 200, 300],
        })

        expenses = calculate_total_expenses(df)
        assert expenses == 1000

    def test_net_savings_calculation(self):
        """Test net savings calculation."""
        df = pd.DataFrame({
            'credit': [50000, 10000],
            'debit': [5000, 3000],
        })

        savings = calculate_net_savings(df)
        assert savings == 52000  # 60000 - 8000

    def test_empty_dataframe(self):
        """Test calculations with empty DataFrame."""
        df = pd.DataFrame()

        assert calculate_total_income(df) == 0
        assert calculate_total_expenses(df) == 0
        assert calculate_net_savings(df) == 0

    def test_monthly_breakdown(self):
        """Test monthly breakdown calculation."""
        df = pd.DataFrame({
            'date': [
                '2025-01-01', '2025-01-15',
                '2025-02-01', '2025-02-15',
                '2025-03-01'
            ],
            'credit': [50000, 0, 50000, 0, 50000],
            'debit': [500, 300, 800, 400, 600],
        })

        breakdown = calculate_monthly_breakdown(df)

        assert '2025-01' in breakdown
        assert '2025-02' in breakdown
        assert '2025-03' in breakdown

        assert breakdown['2025-01']['income'] == 50000
        assert breakdown['2025-01']['expenses'] == 800

    def test_category_breakdown(self):
        """Test category breakdown calculation."""
        df = pd.DataFrame({
            'category': ['Food', 'Food', 'Shopping', 'Transport', 'Food'],
            'debit': [500, 300, 2000, 150, 0],
            'credit': [0, 0, 0, 0, 50000],
        })

        breakdown = calculate_category_breakdown(df)

        assert 'Food' in breakdown
        assert 'Shopping' in breakdown
        assert 'Transport' in breakdown

        assert breakdown['Food']['expenses'] == 800
        assert breakdown['Food']['count'] == 3
        assert breakdown['Shopping']['expenses'] == 2000

    def test_unusual_transaction_detection(self):
        """Test unusual transaction detection."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=10),
            'debit': [100, 150, 120, 130, 110, 140, 125, 135, 1000, 145],
            'description': ['Tx'] * 10,
        })

        unusual = detect_unusual_transactions(df, std_dev_multiplier=2.0)

        assert len(unusual) >= 1
        # The 1000 transaction should be flagged as unusual
        assert any(t['amount'] >= 1000 for t in unusual)

    def test_month_over_month_change(self):
        """Test month-over-month change calculation."""
        df = pd.DataFrame({
            'date': [
                '2025-01-01', '2025-01-15',
                '2025-02-01', '2025-02-15',
            ],
            'debit': [1000, 500, 1500, 800],
            'credit': [50000, 0, 50000, 0],
        })

        changes = calculate_month_over_month_change(df)

        assert '2025-02' in changes
        assert changes['2025-02']['previous_month'] == '2025-01'
        assert changes['2025-02']['percent_change'] > 0  # Expenses increased


class TestAnalysisAgent:
    """Tests for the Analysis Agent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = AnalysisAgent()

    def test_full_analysis(self):
        """Test complete analysis workflow."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=10, freq='D'),
            'description': [f'Txn {i}' for i in range(10)],
            'amount': [1000, 500, 2000, 300, 1500, 800, 1200, 600, 900, 1100],
            'debit': [1000, 500, 2000, 300, 1500, 800, 1200, 600, 900, 1100],
            'credit': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        })

        result = self.agent.analyze(df)

        assert isinstance(result, AnalysisResult)
        assert result.transaction_count == 10
        assert result.total_expenses == 9900  # Sum of all debits

    def test_analysis_with_income(self):
        """Test analysis with income transactions."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5, freq='D'),
            'description': ['Salary', 'Txn2', 'Txn3', 'Txn4', 'Txn5'],
            'amount': [50000, 500, 300, 200, 100],
            'debit': [0, 500, 300, 200, 100],
            'credit': [50000, 0, 0, 0, 0],
        })

        result = self.agent.analyze(df)

        assert result.total_income == 50000
        assert result.total_expenses == 1100
        assert result.net_savings == 48900
        assert result.savings_rate > 90  # High savings rate

    def test_analysis_with_categories(self):
        """Test analysis with categorized transactions."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=6, freq='D'),
            'description': ['Food', 'Food', 'Shopping', 'Transport', 'Food', 'Entertainment'],
            'amount': [500, 300, 2000, 150, 250, 500],
            'debit': [500, 300, 2000, 150, 250, 500],
            'credit': [0] * 6,
            'category': ['Food and Dining', 'Food and Dining', 'Shopping',
                        'Transportation', 'Food and Dining', 'Entertainment'],
        })

        result = self.agent.analyze(df)

        assert result.category_breakdown is not None
        assert 'Food and Dining' in result.category_breakdown
        assert 'Shopping' in result.category_breakdown

        food_breakdown = result.category_breakdown['Food and Dining']
        assert food_breakdown['expenses'] == 1050  # 500 + 300 + 250
        assert food_breakdown['count'] == 3

    def test_highest_spending_category(self):
        """Test highest spending category detection."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=4, freq='D'),
            'description': ['Food', 'Shopping', 'Transport', 'Entertainment'],
            'amount': [500, 2000, 150, 300],
            'debit': [500, 2000, 150, 300],
            'credit': [0] * 4,
            'category': ['Food and Dining', 'Shopping', 'Transportation', 'Entertainment'],
        })

        result = self.agent.analyze(df)

        assert result.highest_spending_category == "Shopping"
        assert result.highest_category_amount == 2000

    def test_empty_dataframe_analysis(self):
        """Test analysis with empty DataFrame."""
        df = pd.DataFrame()

        result = self.agent.analyze(df)

        assert result.transaction_count == 0
        assert result.total_income == 0
        assert result.total_expenses == 0
        assert result.net_savings == 0

    def test_get_monthly_data(self):
        """Test monthly data extraction for charting."""
        df = pd.DataFrame({
            'date': [
                '2025-01-01', '2025-01-15',
                '2025-02-01', '2025-02-15',
            ],
            'credit': [50000, 0, 50000, 0],
            'debit': [1000, 500, 1500, 800],
        })

        self.agent.transactions_df = df
        monthly = self.agent.get_monthly_data()

        assert not monthly.empty
        assert 'month' in monthly.columns
        assert 'income' in monthly.columns
        assert 'expenses' in monthly.columns

    def test_get_category_chart_data(self):
        """Test category chart data extraction."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=4, freq='D'),
            'debit': [500, 2000, 150, 300],
            'credit': [0] * 4,
            'category': ['Food and Dining', 'Shopping', 'Transportation', 'Entertainment'],
        })

        self.agent.transactions_df = df
        self.agent.last_analysis = AnalysisResult(
            total_income=0,
            total_expenses=3050,
            net_savings=-3050,
            savings_rate=0,
            transaction_count=4,
            date_range="2025-01-01 to 2025-01-04",
            monthly_breakdown={},
            category_breakdown={
                'Food and Dining': {'expenses': 500, 'income': 0, 'count': 1, 'total': 500},
                'Shopping': {'expenses': 2000, 'income': 0, 'count': 1, 'total': 2000},
                'Transportation': {'expenses': 150, 'income': 0, 'count': 1, 'total': 150},
                'Entertainment': {'expenses': 300, 'income': 0, 'count': 1, 'total': 300},
            },
            highest_spending_category="Shopping",
            highest_category_amount=2000,
            average_daily_expenditure=762.5,
            month_over_month_changes={},
            unusual_transactions=[],
            recent_transactions=[],
        )

        categories, amounts = self.agent.get_category_chart_data()

        assert 'Shopping' in categories
        assert 2000 in amounts

    def test_search_transactions(self):
        """Test transaction search functionality."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5, freq='D'),
            'description': ['Food Order', 'Shopping', 'Uber Ride', 'Amazon Purchase', 'Food Item'],
            'amount': [500, 2000, 150, 3000, 200],
            'debit': [500, 2000, 150, 3000, 200],
            'credit': [0] * 5,
        })

        self.agent.transactions_df = df
        self.agent.last_analysis = self.agent.analyze(df)

        results = self.agent.search_transactions('Food')

        assert len(results) == 2
        assert all('Food' in r['description'] for r in results)

    def test_filter_transactions(self):
        """Test transaction filtering."""
        from utils.calculations import filter_transactions

        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=10, freq='D'),
            'description': [f'Txn {i}' for i in range(10)],
            'amount': list(range(100, 1100, 100)),
            'debit': list(range(100, 1100, 100)),
            'credit': [0] * 10,
            'category': ['Food'] * 5 + ['Shopping'] * 5,
        })

        filtered = filter_transactions(
            df,
            min_amount=500,
            category='Shopping'
        )

        assert len(filtered) == 5  # Only shopping transactions >= 500

    def test_get_recent_transactions(self):
        """Test recent transactions retrieval."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=20, freq='D'),
            'description': [f'Txn {i}' for i in range(20)],
            'amount': list(range(100, 2100, 100)),
            'debit': list(range(100, 2100, 100)),
            'credit': [0] * 20,
        })

        recent = get_recent_transactions(df, limit=5)

        assert len(recent) == 5
        # Should be sorted by date descending
        assert recent.iloc[0]['date'] > recent.iloc[-1]['date']

    def test_analysis_summary(self):
        """Test analysis summary generation."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5, freq='D'),
            'debit': [100, 200, 150, 300, 250],
            'credit': [50000, 0, 0, 0, 0],
        })

        result = self.agent.analyze(df)
        summary = self.agent.get_analysis_summary()

        assert summary['total_income'] == 50000
        assert summary['total_expenses'] == 1000
        assert summary['transaction_count'] == 5


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_pipeline(self):
        """Test complete data processing pipeline."""
        # Create sample data
        csv_content = b"""Date,Description,Debit,Credit,Amount
2025-01-01,Salary,0,50000,50000
2025-01-02,Food,500,0,500
2025-01-03,Shopping,2000,0,2000
2025-01-04,Transport,150,0,150
2025-01-05,Entertainment,500,0,500"""

        from agents.parser_agent import ParserAgent
        from agents.categorizer_agent import CategorizerAgent
        from agents.analysis_agent import AnalysisAgent

        # Parse
        parser = ParserAgent()
        parse_result = parser.parse_csv(csv_content, "test.csv")

        assert parse_result['success'] is True
        assert len(parse_result['transactions']) == 5

        # Categorize
        categorizer = CategorizerAgent()
        categorized = categorizer.apply_rule_based_categories(parse_result['transactions'])

        # Analyze
        df = pd.DataFrame(categorized)
        analysis_agent = AnalysisAgent()
        analysis = analysis_agent.analyze(df)

        assert analysis.transaction_count == 5
        assert analysis.total_income == 50000
        assert analysis.total_expenses == 3150

    def test_sample_data_analysis(self):
        """Test analysis of included sample data."""
        import os

        sample_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'data', 'sample_transactions.csv'
        )

        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            # Ensure date column exists
            if 'date' not in df.columns:
                df['date'] = pd.date_range('2025-01-01', periods=len(df))

            agent = AnalysisAgent()
            result = agent.analyze(df)

            assert result.transaction_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
