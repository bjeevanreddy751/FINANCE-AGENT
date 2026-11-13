"""
CSV Statement Processor Module
Processes bank statement CSV files and extracts transaction data.
"""

import csv
import io
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import re

from utils.transaction_utils import normalize_date, validate_transaction


class CSVProcessor:
    """
    Processes bank statement CSV files.
    Handles various CSV formats and column naming conventions.
    """

    # Common column name mappings
    COLUMN_MAPPINGS = {
        'date': ['date', 'transaction date', 'trans date', 'txn date',
                 'posted date', 'value date', 'val date', 'domiciled date'],
        'description': ['description', 'narration', 'particulars', 'details',
                        'transaction description', 'remarks', 'memo', 'beneficiary'],
        'debit': ['debit', 'debit amount', 'withdrawal', 'dr', 'debit_amt',
                  'debit amount (rs)', 'amount debited'],
        'credit': ['credit', 'credit amount', 'deposit', 'cr', 'credit_amt',
                   'credit amount (rs)', 'amount credited'],
        'amount': ['amount', 'transaction amount', 'txn amount', 'total amount',
                   'transaction_amt', 'amount'],
        'balance': ['balance', 'closing balance', 'available balance',
                   'account balance', 'running balance', 'bal'],
    }

    def __init__(self):
        self.column_mapping = {}
        self.date_format = None

    def process_csv(self, csv_content: bytes, source: str = "upload") -> pd.DataFrame:
        """
        Process CSV content and extract transactions.

        Args:
            csv_content: Raw CSV content as bytes
            source: Source of the CSV (for error messages)

        Returns:
            DataFrame with standardized transaction data

        Raises:
            ValueError: If CSV processing fails
        """
        try:
            # Handle encoding
            try:
                csv_text = csv_content.decode('utf-8')
            except UnicodeDecodeError:
                csv_text = csv_content.decode('latin-1')

            # Read CSV
            df = pd.read_csv(io.StringIO(csv_text))

            # Normalize column names
            df.columns = self._normalize_column_names(df.columns)

            # Detect column mappings
            self._detect_column_mappings(df.columns)

            # Extract and standardize transactions
            transactions = self._extract_transactions(df)

            if not transactions:
                raise ValueError(f"No valid transactions found in {source} file")

            result_df = pd.DataFrame(transactions)

            # Ensure required columns
            required_cols = ['date', 'description', 'amount']
            for col in required_cols:
                if col not in result_df.columns:
                    result_df[col] = None

            return result_df

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to process CSV file: {str(e)}")

    def process_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Process a CSV file from disk.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with standardized transaction data
        """
        with open(file_path, 'rb') as f:
            content = f.read()
        return self.process_csv(content, f"file: {file_path}")

    def _normalize_column_names(self, columns: List[str]) -> List[str]:
        """Normalize column names to lowercase with underscores."""
        normalized = []
        for col in columns:
            # Convert to lowercase, replace spaces/special chars with underscore
            col_lower = col.lower().strip()
            col_clean = re.sub(r'[^a-z0-9]+', '_', col_lower)
            col_clean = col_clean.strip('_')
            normalized.append(col_clean)
        return normalized

    def _detect_column_mappings(self, columns: List[str]) -> Dict[str, str]:
        """Detect which columns correspond to standard transaction fields."""
        self.column_mapping = {}

        # Columns are normalized to snake_case, so aliases must be too
        # ("Transaction Date" -> "transaction_date").
        normalized_aliases = {
            target: {
                re.sub(r'[^a-z0-9]+', '_', alias.lower()).strip('_')
                for alias in aliases
            }
            for target, aliases in self.COLUMN_MAPPINGS.items()
        }

        for col in columns:
            for target, aliases in normalized_aliases.items():
                if col in aliases:
                    self.column_mapping[target] = col
                    break

        return self.column_mapping

    def _extract_transactions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract transaction data from DataFrame.
        Standardizes to: date, description, debit, credit, amount
        """
        transactions = []

        # Get column mappings with fallback
        date_col = self.column_mapping.get('date')
        desc_col = self.column_mapping.get('description')
        debit_col = self.column_mapping.get('debit')
        credit_col = self.column_mapping.get('credit')
        amount_col = self.column_mapping.get('amount')
        balance_col = self.column_mapping.get('balance')

        for idx, row in df.iterrows():
            transaction = {}

            # Extract date
            if date_col and date_col in df.columns:
                date_val = row[date_col]
                if pd.notna(date_val):
                    normalized_date = normalize_date(str(date_val))
                    if normalized_date:
                        transaction['date'] = normalized_date

            # Extract description
            if desc_col and desc_col in df.columns:
                desc_val = row[desc_col]
                if pd.notna(desc_val):
                    transaction['description'] = str(desc_val).strip()

            # Extract debit amount
            if debit_col and debit_col in df.columns:
                debit_val = row[debit_col]
                if pd.notna(debit_val):
                    transaction['debit'] = self._parse_amount(debit_val)

            # Extract credit amount
            if credit_col and credit_col in df.columns:
                credit_val = row[credit_col]
                if pd.notna(credit_val):
                    transaction['credit'] = self._parse_amount(credit_val)

            # Extract total amount
            if amount_col and amount_col in df.columns:
                amount_val = row[amount_col]
                if pd.notna(amount_val):
                    transaction['amount'] = self._parse_amount(amount_val)

            # Extract balance if available
            if balance_col and balance_col in df.columns:
                balance_val = row[balance_col]
                if pd.notna(balance_val):
                    transaction['balance'] = self._parse_amount(balance_val)

            # Skip if no data
            if not transaction:
                continue

            # Calculate amount if not present but debit/credit exist
            if 'amount' not in transaction or transaction.get('amount') is None:
                debit = transaction.get('debit') or 0
                credit = transaction.get('credit') or 0
                transaction['amount'] = debit + credit

            # Ensure required fields
            if not transaction.get('amount') or transaction['amount'] == 0:
                continue

            # Default description if missing
            if not transaction.get('description'):
                transaction['description'] = "Transaction"

            transactions.append(transaction)

        return transactions

    def _parse_amount(self, value: Any) -> float:
        """Parse amount value to float."""
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove currency symbols and whitespace
            cleaned = re.sub(r'[₹$€£,\s]', '', value.strip())

            # Handle negative amounts in parentheses
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]

            try:
                return float(cleaned)
            except ValueError:
                return 0.0

        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def detect_csv_columns(self, csv_content: bytes) -> Dict[str, str]:
        """
        Detect column names in a CSV file.

        Args:
            csv_content: Raw CSV content

        Returns:
            Dictionary mapping standard names to actual column names
        """
        try:
            csv_text = csv_content.decode('utf-8')
        except UnicodeDecodeError:
            csv_text = csv_content.decode('latin-1')

        df = pd.read_csv(io.StringIO(csv_text))
        df.columns = self._normalize_column_names(df.columns)
        return self._detect_column_mappings(df.columns)


def process_csv_transactions(csv_content: bytes) -> pd.DataFrame:
    """
    Convenience function to process CSV transactions.

    Args:
        csv_content: Raw CSV content as bytes

    Returns:
        DataFrame with standardized transaction data
    """
    processor = CSVProcessor()
    return processor.process_csv(csv_content)


def detect_csv_columns(csv_content: bytes) -> Dict[str, str]:
    """
    Detect column names in a CSV file.

    Args:
        csv_content: Raw CSV content

    Returns:
        Dictionary mapping standard names to actual column names
    """
    processor = CSVProcessor()
    return processor.detect_csv_columns(csv_content)
