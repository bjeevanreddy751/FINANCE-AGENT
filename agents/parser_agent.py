"""
Statement Parser Agent
Extracts and processes transactions from PDF and CSV bank statements.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import io

from utils.pdf_extractor import PDFExtractor, extract_pdf_transactions
from utils.csv_processor import CSVProcessor, process_csv_transactions
from utils.transaction_utils import (
    validate_transaction,
    detect_invalid_transactions,
    merge_transactions,
    format_transaction_for_display
)
from prompts.parser_prompt import get_parser_prompt


class ParserAgent:
    """
    Agent responsible for extracting transactions from bank statements.

    Capabilities:
    - PDF statement extraction
    - CSV statement processing
    - Data validation and cleaning
    - Transaction deduplication
    """

    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.csv_processor = CSVProcessor()
        self.transactions: List[Dict] = []
        self.extraction_stats = {}

    def parse_pdf(self, pdf_content: bytes, filename: str = "statement.pdf") -> Dict:
        """
        Parse a PDF bank statement.

        Args:
            pdf_content: PDF file content as bytes
            filename: Name of the PDF file

        Returns:
            Dictionary with extraction results
        """
        result = {
            'success': False,
            'transactions': [],
            'error': None,
            'stats': {}
        }

        try:
            # Save to temp file for extraction
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name

            try:
                # Extract transactions
                df = self.pdf_extractor.extract_from_pdf(tmp_path)

                if df.empty:
                    result['error'] = self._explain_empty_pdf(self.pdf_extractor.last_diagnostics)
                    return result

                # Convert to list of dicts
                transactions = self._df_to_transactions(df)

                # Validate transactions
                valid, invalid = self._validate_transactions(transactions)

                result['success'] = True
                result['transactions'] = valid
                result['invalid_transactions'] = invalid
                result['stats'] = {
                    'total_found': len(transactions),
                    'valid': len(valid),
                    'invalid': len(invalid),
                    'source': f"PDF: {filename}",
                    'date_range': self._get_date_range(valid)
                }

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            result['error'] = f"Failed to parse PDF: {str(e)}"

        return result

    def parse_csv(self, csv_content: bytes, filename: str = "statement.csv") -> Dict:
        """
        Parse a CSV bank statement.

        Args:
            csv_content: CSV file content as bytes
            filename: Name of the CSV file

        Returns:
            Dictionary with extraction results
        """
        result = {
            'success': False,
            'transactions': [],
            'error': None,
            'stats': {}
        }

        try:
            # Process CSV
            df = self.csv_processor.process_csv(csv_content, filename)

            if df.empty:
                result['error'] = "No transactions found in the CSV file."
                return result

            # Convert to list of dicts
            transactions = self._df_to_transactions(df)

            # Validate transactions
            valid, invalid = self._validate_transactions(transactions)

            result['success'] = True
            result['transactions'] = valid
            result['invalid_transactions'] = invalid
            result['stats'] = {
                'total_found': len(transactions),
                'valid': len(valid),
                'invalid': len(invalid),
                'source': f"CSV: {filename}",
                'date_range': self._get_date_range(valid)
            }

        except Exception as e:
            result['error'] = f"Failed to parse CSV: {str(e)}"

        return result

    def parse_file(self, file_bytes: bytes, file_type: str,
                   filename: str = "statement") -> Dict:
        """
        Parse a file of the specified type.

        Args:
            file_bytes: File content as bytes
            file_type: 'pdf' or 'csv'
            filename: Name of the file

        Returns:
            Dictionary with parsing results
        """
        if file_type.lower() == 'pdf':
            return self.parse_pdf(file_bytes, filename)
        elif file_type.lower() == 'csv':
            return self.parse_csv(file_bytes, filename)
        else:
            return {
                'success': False,
                'transactions': [],
                'error': f"Unsupported file type: {file_type}",
                'stats': {}
            }

    def merge_transactions(self, new_transactions: List[Dict]) -> int:
        """
        Merge new transactions into existing collection.

        Args:
            new_transactions: New transactions to merge

        Returns:
            Number of new transactions added
        """
        before_count = len(self.transactions)
        self.transactions = merge_transactions(self.transactions, new_transactions)
        added = len(self.transactions) - before_count

        self._update_stats()
        return added

    def get_transactions_df(self) -> pd.DataFrame:
        """
        Get all transactions as a DataFrame.

        Returns:
            DataFrame with all transactions
        """
        if not self.transactions:
            return pd.DataFrame(columns=['date', 'description', 'amount', 'debit', 'credit'])

        df = pd.DataFrame(self.transactions)
        return df

    def get_transaction_count(self) -> int:
        """Get the total number of parsed transactions."""
        return len(self.transactions)

    def clear_transactions(self):
        """Clear all parsed transactions."""
        self.transactions = []
        self.extraction_stats = {}

    def _df_to_transactions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert DataFrame to list of transaction dictionaries.

        Args:
            df: DataFrame with transaction data

        Returns:
            List of transaction dictionaries
        """
        transactions = []

        for _, row in df.iterrows():
            txn = {
                'date': row.get('date'),
                'description': str(row.get('description', '')) if pd.notna(row.get('description')) else '',
                'amount': float(row.get('amount', 0)) if pd.notna(row.get('amount')) else 0,
                'debit': float(row.get('debit', 0)) if pd.notna(row.get('debit')) else 0,
                'credit': float(row.get('credit', 0)) if pd.notna(row.get('credit')) else 0,
            }

            # Add balance if available
            if 'balance' in df.columns and pd.notna(row.get('balance')):
                txn['balance'] = float(row['balance'])

            # Validate the transaction
            txn = validate_transaction(txn)

            if txn.get('amount', 0) > 0:
                transactions.append(txn)

        return transactions

    def _validate_transactions(self, transactions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate and separate valid from invalid transactions.

        Args:
            transactions: List of transactions

        Returns:
            Tuple of (valid_transactions, invalid_transactions)
        """
        valid = []
        invalid = []

        for txn in transactions:
            # Check for required fields
            if not txn.get('date') or not txn.get('description'):
                invalid.append({**txn, 'validation_issue': 'Missing required fields'})
                continue

            if txn.get('amount', 0) <= 0:
                invalid.append({**txn, 'validation_issue': 'Invalid or zero amount'})
                continue

            # Check for suspicious patterns
            issues = detect_invalid_transactions([txn], min_amount=0.01)
            if issues:
                invalid.append({**txn, 'validation_issues': issues[0].get('validation_issues', [])})
            else:
                valid.append(txn)

        return valid, invalid

    def _explain_empty_pdf(self, diagnostics: Optional[Dict] = None) -> str:
        """Build an actionable message when a PDF yields no transactions."""
        diag = diagnostics or {}
        pages = diag.get('pages', 0)
        pages_with_text = diag.get('pages_with_text', 0)
        chars = diag.get('text_chars', 0)

        if pages and not pages_with_text:
            return (
                "No transactions could be extracted from the PDF because it contains no "
                "selectable text (it looks like a scanned or image-only statement). "
                "FinanceIQ reads text-based PDFs; please upload a downloaded statement PDF "
                "from your bank, or export it as CSV."
            )

        if pages_with_text and chars:
            dates = diag.get('lines_with_date', 0)
            amounts = diag.get('lines_with_amount', 0)
            return (
                "No transactions could be extracted from the PDF even though text was read "
                f"from {pages_with_text} of {pages} page(s) ({chars} characters). "
                f"Lines containing a date: {dates}; lines containing an amount: {amounts}. "
                "FinanceIQ expects one transaction per line, e.g. "
                "'01/04/2025  Swiggy Order  Rs.450.00'. If your statement puts each column on "
                "its own line or uses an unusual date format (e.g. 01-Apr-2025), export the "
                "statement as CSV instead."
            )

        return (
            "No transactions could be extracted from the PDF. The file could be empty, "
            "corrupted, or password protected. Try a CSV export of the same statement."
        )

    def _get_date_range(self, transactions: List[Dict]) -> str:
        """Get the date range of transactions."""
        if not transactions:
            return "N/A"

        dates = [txn['date'] for txn in transactions if txn.get('date')]
        if not dates:
            return "N/A"

        return f"{min(dates)} to {max(dates)}"

    def _update_stats(self):
        """Update extraction statistics."""
        self.extraction_stats = {
            'total_transactions': len(self.transactions),
            'date_range': self._get_date_range(self.transactions),
            'total_amount': sum(t.get('amount', 0) for t in self.transactions),
            'sources': list(set(t.get('_source', 'unknown') for t in self.transactions))
        }

    def get_stats(self) -> Dict:
        """Get extraction statistics."""
        return self.extraction_stats.copy() if self.extraction_stats else {
            'total_transactions': len(self.transactions),
            'date_range': self._get_date_range(self.transactions)
        }


# Convenience function
def parse_statement(file_bytes: bytes, file_type: str,
                    filename: str = "statement") -> Dict:
    """
    Parse a bank statement file.

    Args:
        file_bytes: File content
        file_type: 'pdf' or 'csv'
        filename: File name

    Returns:
        Parsing result dictionary
    """
    agent = ParserAgent()
    return agent.parse_file(file_bytes, file_type, filename)


# For LCEL integration
from langchain_core.tools import tool

@tool
def parse_bank_statement(file_bytes: str, file_type: str, filename: str) -> str:
    """
    Parse a bank statement file (PDF or CSV).

    Args:
        file_bytes: Base64 encoded file content
        file_type: 'pdf' or 'csv'
        filename: Name of the file

    Returns:
        JSON string with parsing results
    """
    import base64
    import json

    try:
        content = base64.b64decode(file_bytes)
    except Exception:
        content = file_bytes.encode() if isinstance(file_bytes, str) else file_bytes

    agent = ParserAgent()
    result = agent.parse_file(content, file_type, filename)

    return json.dumps(result, indent=2, default=str)
