"""
PDF Statement Extractor Module
Extracts transaction data from PDF bank statements.
"""

import re
from typing import List, Dict, Optional, Any
from datetime import datetime
from pypdf import PdfReader
import pandas as pd

from utils.transaction_utils import normalize_date, validate_transaction
from utils.csv_processor import detect_csv_columns


class PDFExtractor:
    """
    Extracts transaction data from PDF bank statements.
    Handles various PDF table formats and extracts date, description, amounts.
    """

    def __init__(self):
        # Common patterns for transaction dates (DD/MM/YYYY, DD-MM-YYYY, etc.)
        self.date_patterns = [
            r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
            r'\b(\d{1,2}-\d{1,2}-\d{2,4})\b',
            r'\b(\d{4}/\d{1,2}/\d{1,2})\b',
            r'\b(\d{4}-\d{1,2}-\d{1,2})\b',
            r'\b(\d{2,4}\.\d{1,2}\.\d{1,2})\b',
        ]

        # Patterns for amounts: 1234.50 / 1,234.50 / (1,234.50) / 1,234.50 Cr / Rs.1,234
        # Matches plain numbers of any length too, so "1234.50" is not cut down to "123".
        self.amount_pattern = (
            r'(?:₹|rs\.?|inr|\$|€|£)?\s*\(?\s*'
            r'(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)'
            r'\s*\)?\s*(?:cr|dr|credit|debit)?'
        )

        # Patterns for debit/credit indicators
        self.debit_indicators = ['dr', 'debit', 'withdrawal', 'payment', 'purchase', 'debit']
        self.credit_indicators = ['cr', 'credit', 'deposit', 'salary', 'credit', 'received']

        # Filled in on every extraction run so the caller can explain empty results.
        self.last_diagnostics: Dict[str, Any] = {}
        self._line_diagnostics: Dict[str, int] = {}

    def extract_from_pdf(self, pdf_path: str) -> pd.DataFrame:
        """
        Extract transactions from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            DataFrame with extracted transactions
        """
        try:
            reader = PdfReader(pdf_path)
            all_text = []
            pages_with_text = 0

            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                except Exception:
                    text = None
                if text and text.strip():
                    pages_with_text += 1
                    all_text.append(text)
                    # Also try to extract tables if available
                    tables = self._extract_tables_from_page(page)
                    for table in tables:
                        all_text.append(self._tables_to_text(table))

            full_text = '\n'.join(all_text)
            df = self._parse_text_to_transactions(full_text)

            self.last_diagnostics = {
                'pages': len(reader.pages),
                'pages_with_text': pages_with_text,
                'text_chars': len(full_text.strip()),
                'transactions_found': len(df),
                **self._line_diagnostics
            }

            return df

        except Exception as e:
            raise ValueError(f"Failed to extract from PDF: {str(e)}")

    def _extract_tables_from_page(self, page) -> List[List[List[str]]]:
        """Try to extract tables from a PDF page."""
        try:
            # pypdf doesn't have built-in table extraction
            # We'll handle this through text extraction
            return []
        except Exception:
            return []

    def _tables_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to text format."""
        lines = []
        for row in table:
            lines.append(' | '.join(row))
        return '\n'.join(lines)

    def _parse_text_to_transactions(self, text: str) -> pd.DataFrame:
        """
        Parse extracted text to find transaction data.
        Uses regex patterns to identify dates, descriptions, and amounts.
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        lines = self._merge_split_rows(lines)
        transactions = []

        # Record what the document contained, so an empty result can be explained.
        self._line_diagnostics = {
            'lines_with_date': sum(1 for l in lines if self._line_has_date(l)),
            'lines_with_amount': sum(1 for l in lines if self._line_has_amount(l)),
        }

        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # A line whose only numbers belong to a date is a header, not a transaction.
            if not self._line_has_amount(line):
                continue

            # Try to parse multi-column format (date, description, amount)
            parsed = self._try_parse_transaction_line(line)
            if parsed:
                transactions.append(parsed)
            else:
                # Try to find all dates and amounts in the line
                parsed = self._try_parse_complex_line(line)
                if parsed:
                    transactions.append(parsed)

        return self._create_dataframe(transactions)

    def _try_parse_transaction_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse a line as a simple transaction entry.
        Expected formats:
        - Date Description Amount
        - Date,Description,Amount
        - |Date|Description|Amount|
        """
        # Split by common delimiters
        parts = re.split(r'\s{2,}|\t+', line)

        if len(parts) >= 3:
            # Try to identify parts
            date_part = None
            amount_part = None
            desc_parts = []

            for part in parts:
                part = part.strip().strip('|')

                # Check if it's a date
                if self._is_date(part):
                    date_part = part
                # Check if it's an amount
                elif self._is_amount(part):
                    amount_part = part
                else:
                    desc_parts.append(part)

            if date_part and amount_part:
                description = ' '.join(desc_parts).strip()
                # Narration cells often still carry the amount and a Dr/Cr marker.
                if description:
                    description = re.sub(r'(?i)\b(?:dr|cr)\b', ' ', self._strip_dates(description))
                    for token in re.findall(self.amount_pattern, description):
                        description = description.replace(token, ' ', 1)
                    description = re.sub(r'\s+', ' ', description).strip()
                date_val = self._parse_date(date_part)
                amounts = self._parse_amounts(amount_part)

                # Dr/Cr markers can sit in the narration, not only next to the amount.
                debit_val, credit_val = self._split_direction(
                    line, amounts.get('debit'), amounts.get('credit'), amounts.get('total')
                )

                return {
                    'date': date_val,
                    'description': description,
                    'debit': debit_val,
                    'credit': credit_val,
                    'amount': debit_val or credit_val
                }

        # Try pipe-separated or space-separated with amounts
        return self._try_parse_with_amounts(line)

    def _try_parse_complex_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Try to parse a line with scattered date and amount information."""
        dates_found = []
        amounts_found = []

        # Find all dates
        for pattern in self.date_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                parsed_date = self._parse_date(match)
                if parsed_date:
                    dates_found.append(parsed_date)

        # Find all amounts outside the date tokens
        amount_matches = re.findall(self.amount_pattern, self._strip_dates(line))
        for match in amount_matches:
            try:
                amount_val = float(match.replace(',', ''))
                amounts_found.append(amount_val)
            except ValueError:
                continue

        if dates_found and amounts_found:
            date_val = dates_found[0]  # Use first date found

            # Determine debit/credit from Dr/Cr markers, defaulting to a debit
            # (most statement rows are payments).
            total = amounts_found[0]
            debit_val, credit_val = self._split_direction(line, None, None, total)

            # Extract description (remove dates and amounts)
            desc = self._extract_description(line, dates_found, amounts_found)

            return {
                'date': date_val,
                'description': desc,
                'debit': debit_val,
                'credit': credit_val,
                'amount': debit_val or credit_val or amounts_found[0] if amounts_found else None
            }

        return None

    def _try_parse_with_amounts(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a line focusing on extracting amounts and associated dates."""
        # Only look for amounts outside the date, otherwise "06/04/2025" reads as 6.
        without_dates = self._strip_dates(line)
        amount_tokens = re.findall(self.amount_pattern, without_dates)
        amounts = [self._to_number(token) for token in amount_tokens]
        amounts = [a for a in amounts if a]

        if not amounts:
            return None

        date_val = None
        for pattern in self.date_patterns:
            match = re.search(pattern, line)
            if match:
                date_val = self._parse_date(match.group(1))
                break

        # Extract description: drop dates, amounts and Dr/Cr markers.
        description = re.sub(r'(?i)\b(?:dr|cr)\b', ' ', without_dates)
        for token in amount_tokens:
            description = description.replace(token, ' ', 1)
        description = re.sub(r'\s+', ' ', description)
        description = description.strip().strip('|').strip('-').strip()
        if not description:
            description = "Transaction"

        debit_val, credit_val = self._split_direction(line, None, None, amounts[0])

        return {
            'date': date_val,
            'description': description,
            'debit': debit_val,
            'credit': credit_val,
            'amount': max(debit_val or 0, credit_val or 0)
        }

    def _merge_split_rows(self, lines: List[str]) -> List[str]:
        """
        Join rows whose date and amount ended up on separate lines.

        Many PDF statements wrap a transaction over two lines: the first holds the
        date and narration, the second the amount. Such rows are otherwise dropped.
        """
        merged: List[str] = []
        pending: Optional[str] = None

        for line in lines:
            if pending is None:
                pending = line
                continue

            pending_has_date = self._line_has_date(pending)
            pending_has_amount = self._line_has_amount(pending)
            line_has_amount = self._line_has_amount(line)
            line_has_date = self._line_has_date(line)

            if pending_has_date and not pending_has_amount and line_has_amount and not line_has_date:
                merged.append(f"{pending} {line}")
                pending = None
            else:
                merged.append(pending)
                pending = line

        if pending:
            merged.append(pending)

        return merged

    def _line_has_date(self, line: str) -> bool:
        """Check whether a text line contains any recognizable date."""
        return any(re.search(pattern, line) for pattern in self.date_patterns)

    def _line_has_amount(self, line: str) -> bool:
        """Check whether a text line contains an amount that is not part of a date."""
        return any(
            self._to_number(match) is not None
            for match in re.findall(self.amount_pattern, self._strip_dates(line))
        )

    def _strip_dates(self, line: str) -> str:
        """Remove date tokens so their digits are not mistaken for amounts."""
        stripped = line
        for pattern in self.date_patterns:
            stripped = re.sub(pattern, ' ', stripped)
        return stripped

    def _split_direction(self, line: str, debit: Optional[float],
                         credit: Optional[float],
                         total: Optional[float]) -> tuple:
        """
        Decide whether an amount is a debit or a credit.

        Dr/Cr/credit/debit markers anywhere in the row take precedence over an
        already-guessed amount. Rows with no marker are treated as debits, since
        most statement rows are payments.
        """
        has_debit = bool(re.search(r'(?i)\b(?:dr|debit|withdrawal|paid|payment|purchase)\b', line))
        has_credit = bool(re.search(r'(?i)\b(?:cr|credit|deposit|salary|received|refund)\b', line))

        if total is None:
            total = (debit or 0) + (credit or 0) or debit or credit

        if has_debit and not has_credit:
            return total, None
        if has_credit and not has_debit:
            return None, total

        if debit or credit:
            return debit, credit

        return total, None

    def _to_number(self, text: str) -> Optional[float]:
        """
        Convert a statement amount token to a float.

        Handles currency prefixes (Rs., INR, symbols), thousands separators,
        accounting-style negatives in parentheses, and Cr/Dr suffixes.
        Returns None when the token is not an amount.
        """
        if text is None:
            return None

        token = str(text).strip().strip('|').strip()
        if not token:
            return None

        negative = token.startswith('(') and token.endswith(')')
        token = token.strip('()').strip()
        token = re.sub(r'(?i)\b(?:rs\.?|inr|cr|dr|credit|debit)\b', '', token)
        for symbol in ('₹', '$', '€', '£'):
            token = token.replace(symbol, '')
        token = token.replace(',', '').replace(' ', '')

        if not re.fullmatch(r'-?\d+(?:\.\d{1,2})?', token):
            return None

        value = float(token)
        return -value if negative else value

    def _is_date(self, text: str) -> bool:
        """Check if text looks like a date."""
        text = text.strip()
        for pattern in self.date_patterns:
            if re.match(pattern, text):
                if self._parse_date(text):
                    return True
        return False

    def _is_amount(self, text: str) -> bool:
        """Check if text looks like an amount."""
        return self._to_number(text) is not None

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse a date string to YYYY-MM-DD format."""
        date_str = date_str.strip()

        for fmt in ['%d/%m/%Y', '%d/%m/%y', '%d-%m-%Y', '%d-%m-%y',
                    '%m/%d/%Y', '%m/%d/%y', '%d.%m.%Y', '%d.%m.%y',
                    '%Y/%m/%d', '%Y-%m-%d']:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue

        return None

    def _parse_amounts(self, text: str) -> Dict[str, Optional[float]]:
        """Parse debit and credit amounts from text."""
        result = {'debit': None, 'credit': None, 'total': None}

        amounts = re.findall(self.amount_pattern, text)
        amounts = [float(a.replace(',', '')) for a in amounts if a]

        if not amounts:
            return result

        # Try to determine debit vs credit
        text_lower = text.lower()
        if 'debit' in text_lower or 'dr' in text_lower:
            result['debit'] = amounts[0] if amounts else None
        elif 'credit' in text_lower or 'cr' in text_lower:
            result['credit'] = amounts[0] if amounts else None
        elif len(amounts) >= 2:
            # Two amounts: first is usually debit, second is credit or balance
            result['debit'] = amounts[0]
            result['credit'] = amounts[1]
        else:
            # Single amount - check context
            if any(ind in text_lower for ind in ['withdrawal', 'payment', 'purchase', 'expense']):
                result['debit'] = amounts[0]
            else:
                result['credit'] = amounts[0]

        result['total'] = result['debit'] or result['credit']
        return result

    def _extract_description(self, line: str, dates: List[str], amounts: List[float]) -> str:
        """Extract description by removing dates and amounts."""
        desc = line

        # Remove dates
        for d in dates:
            desc = desc.replace(str(d), '', 1)

        # Remove amounts
        for a in amounts:
            desc = desc.replace(str(a), '', 1)
            desc = desc.replace(f"₹{a}", '', 1)
            desc = desc.replace(f"Rs.{a}", '', 1)

        # Clean up description
        desc = re.sub(r'\s+', ' ', desc)
        desc = re.sub(r'[\|\-_]{2,}', ' ', desc)
        desc = desc.strip().strip('|').strip('-').strip()

        return desc if desc else "Transaction"

    def _create_dataframe(self, transactions: List[Dict]) -> pd.DataFrame:
        """Convert list of transaction dicts to DataFrame."""
        if not transactions:
            return pd.DataFrame(columns=['date', 'description', 'debit', 'credit', 'amount'])

        df = pd.DataFrame(transactions)

        # Ensure all columns exist
        for col in ['date', 'description', 'debit', 'credit', 'amount']:
            if col not in df.columns:
                df[col] = None

        # Convert dates
        if 'date' in df.columns:
            df['date'] = df['date'].apply(lambda x: x if x else None)

        # Fill missing values
        for col in ['debit', 'credit', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Remove duplicate transactions based on date, description, and amount
        df = df.drop_duplicates(subset=['date', 'description', 'amount'], keep='first')

        # Remove transactions with no useful data
        df = df[(df['description'].notna()) & (df['amount'] > 0)]

        return df.reset_index(drop=True)


def extract_pdf_transactions(pdf_path: str) -> pd.DataFrame:
    """
    Convenience function to extract transactions from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        DataFrame with extracted transactions

    Raises:
        ValueError: If PDF extraction fails
    """
    extractor = PDFExtractor()
    return extractor.extract_from_pdf(pdf_path)
