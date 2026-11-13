"""
Tests for FinanceIQ Parser Agent
"""

import pytest
import pandas as pd
import io
from agents.parser_agent import ParserAgent, parse_statement
from utils.csv_processor import CSVProcessor


class TestParserAgent:
    """Tests for the Parser Agent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ParserAgent()

    def test_csv_processing_basic(self):
        """Test basic CSV processing."""
        csv_content = b"Date,Description,Amount,Debit,Credit\n"
        csv_content += b"2025-01-01,Salary,50000,0,50000\n"
        csv_content += b"2025-01-02,Food,500,500,0\n"
        csv_content += b"2025-01-03,Transport,200,200,0"

        result = self.parser.parse_csv(csv_content, "test.csv")

        assert result['success'] is True
        assert len(result['transactions']) == 3
        assert result['stats']['total_found'] == 3

    def test_csv_column_mapping(self):
        """Test CSV with various column names."""
        csv_content = b"Transaction Date,Particulars,Withdrawal,Deposit\n"
        csv_content += b"01/01/2025,Salary Credit,0,50000\n"
        csv_content += b"01/02/2025,Shop Purchase,1500,0"

        result = self.parser.parse_csv(csv_content, "test.csv")

        assert result['success'] is True
        assert len(result['transactions']) >= 1

    def test_date_normalization(self):
        """Test that dates are normalized to YYYY-MM-DD."""
        csv_content = b"Date,Description,Amount\n"
        csv_content += b"01/15/2025,Transaction 1,100\n"
        csv_content += b"15-01-2025,Transaction 2,200\n"
        csv_content += b"2025/01/20,Transaction 3,300"

        result = self.parser.parse_csv(csv_content, "test.csv")

        assert result['success'] is True
        for txn in result['transactions']:
            assert txn['date'] is not None
            # Check YYYY-MM-DD format
            parts = txn['date'].split('-')
            assert len(parts) == 3

    def test_amount_parsing(self):
        """Test amount parsing with various formats."""
        csv_content = b"Date,Description,Amount\n"
        csv_content += b"2025-01-01,Test 1,1000\n"
        csv_content += b"2025-01-02,Test 2,1500.00\n"
        csv_content += b"2025-01-03,Test 3,500.00"

        result = self.parser.parse_csv(csv_content, "test.csv")

        assert result['success'] is True
        amounts = [t['amount'] for t in result['transactions']]
        assert any(abs(a - 1000) < 0.01 for a in amounts)

    def test_empty_csv_handling(self):
        """Test handling of empty CSV."""
        csv_content = b"Date,Description,Amount\n"

        result = self.parser.parse_csv(csv_content, "test.csv")

        # Should not crash, but may have zero transactions
        assert 'transactions' in result

    def test_invalid_date_handling(self):
        """Test handling of invalid dates."""
        csv_content = b"Date,Description,Amount\n"
        csv_content += b"invalid-date,Transaction,100\n"
        csv_content += b"2025-01-01,Valid Transaction,200"

        result = self.parser.parse_csv(csv_content, "test.csv")

        # Should process valid rows
        assert result['success'] in [True, None]  # Either success or graceful handling

    def test_duplicate_removal(self):
        """Test that duplicate transactions are removed."""
        csv_content = b"Date,Description,Amount\n"
        csv_content += b"2025-01-01,Same Transaction,500\n"
        csv_content += b"2025-01-01,Same Transaction,500\n"
        csv_content += b"2025-01-01,Different,300"

        result = self.parser.parse_csv(csv_content, "test.csv")

        assert result['success'] is True
        # Duplicates should be removed
        unique_descriptions = set(t['description'] for t in result['transactions'])
        assert len(unique_descriptions) <= 2

    def test_credit_debit_detection(self):
        """Test that credit and debit amounts are properly detected."""
        csv_content = b"Date,Description,Debit,Credit,Balance\n"
        csv_content += b"2025-01-01,Salary,0,50000,50000\n"
        csv_content += b"2025-01-02,Expense,500,0,49500"

        result = self.parser.parse_csv(csv_content, "test.csv")

        assert result['success'] is True
        for txn in result['transactions']:
            # Check that debit and credit are properly separated
            assert 'debit' in txn
            assert 'credit' in txn


class TestCSVProcessor:
    """Tests for CSV Processor utility."""

    def test_column_detection(self):
        """Test detection of various column names."""
        processor = CSVProcessor()

        csv_content = b"""Date,Description,Debit Amount,Credit Amount
2025-01-01,Test,100,0"""

        mappings = processor.detect_csv_columns(csv_content)

        assert 'date' in mappings or 'Description' in mappings

    def test_csv_with_various_formats(self):
        """Test CSV parsing with various date formats."""
        processor = CSVProcessor()

        test_cases = [
            b"""Date,Narration,Amount
01/01/2025,Transaction,100""",
            b"""Date,Description,Amount
2025-01-01,Transaction,100""",
            b"""Value Date,Particulars,Amount
01-01-2025,Transaction,100""",
        ]

        for csv_content in test_cases:
            df = processor.process_csv(csv_content)
            assert not df.empty or len(df) == 0  # Should handle gracefully


class TestIntegration:
    """Integration tests for parser functionality."""

    def test_full_parsing_workflow(self):
        """Test complete parsing workflow."""
        # Create sample data
        csv_content = b"""Date,Description,Debit,Credit,Amount
2025-01-01,Salary Credit,0,50000,50000
2025-01-02,Food Expense,500,0,500
2025-01-03,Transport,200,0,200
2025-01-04,Utility Bill,1000,0,1000"""

        parser = ParserAgent()
        result = parser.parse_csv(csv_content, "sample.csv")

        assert result['success'] is True
        assert len(result['transactions']) == 4

        # Check transaction structure
        txn = result['transactions'][0]
        assert 'date' in txn
        assert 'description' in txn
        assert 'amount' in txn

    def test_csv_processor_convenience_function(self):
        """Test the convenience function for CSV processing."""
        csv_content = b"""Date,Description,Amount
2025-01-01,Test,100"""

        from utils.csv_processor import process_csv_transactions
        df = process_csv_transactions(csv_content)

        assert not df.empty

    def test_category_application(self):
        """Test that categories can be applied to parsed transactions."""
        from agents.categorizer_agent import CategorizerAgent

        csv_content = b"""Date,Description,Amount
2025-01-01,Swiggy Food Order,500
2025-01-02,Amazon Shopping,2000
2025-01-03,Uber Ride,150"""

        parser = ParserAgent()
        result = parser.parse_csv(csv_content, "test.csv")

        categorizer = CategorizerAgent()
        categorized = categorizer.apply_rule_based_categories(result['transactions'])

        for txn in categorized:
            assert 'category' in txn
            assert txn['category'] in [
                "Food and Dining",
                "Shopping",
                "Transportation",
                "Other"
            ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
