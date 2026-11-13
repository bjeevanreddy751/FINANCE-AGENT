"""
Parser Agent Prompts
Prompt templates for statement parsing and extraction.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# System prompt for transaction parsing
PARSER_SYSTEM_PROMPT = """You are a FinanceIQ Statement Parser Agent.
Your task is to analyze bank statement data and extract structured transaction information.

You specialize in:
- Extracting transaction dates, descriptions, and amounts
- Identifying debit and credit transactions
- Normalizing different date formats to YYYY-MM-DD
- Handling various bank statement formats

Guidelines:
- Always extract the date, description, and amount
- Identify whether it's a debit (expense) or credit (income)
- Be precise with amounts - don't round or approximate
- Flag any suspicious or malformed entries
- If a line is not a transaction (header, footer, advertisement), skip it
"""

# Human prompt for parsing assistance
PARSER_HUMAN_PROMPT = """Please analyze the following bank statement transaction data:

Statement Data:
{statement_data}

Please extract and structure the following for each transaction:
1. Date (in YYYY-MM-DD format)
2. Description (clean transaction description)
3. Amount
4. Type: "debit" for expenses, "credit" for income
5. Notes: any observations about the transaction

Return the data in a structured format that can be easily parsed.
Focus on accuracy and completeness.
"""

# Create the prompt template
parser_prompt_template = ChatPromptTemplate.from_messages([
    ("system", PARSER_SYSTEM_PROMPT),
    ("human", PARSER_HUMAN_PROMPT),
])

# Standalone prompt template for simple extraction
SIMPLE_PARSER_PROMPT = PromptTemplate(
    input_variables=["transaction_line"],
    template="""Analyze this transaction line and extract the key details:

Transaction: {transaction_line}

Extract:
- date: (YYYY-MM-DD format or null)
- description: (clean description)
- debit: (amount if expense, null otherwise)
- credit: (amount if income, null otherwise)
- category_hint: (suggested category based on description)

Return as JSON. If no valid transaction can be extracted, return null.""",
)


def get_parser_prompt(statement_data: str) -> ChatPromptTemplate:
    """
    Get a parser prompt with the provided statement data.

    Args:
        statement_data: Raw statement text or data summary

    Returns:
        Configured ChatPromptTemplate
    """
    return parser_prompt_template.format(statement_data=statement_data)


def get_simple_parser_prompt(transaction_line: str) -> PromptTemplate:
    """
    Get a simple parser prompt for a single transaction line.

    Args:
        transaction_line: Single transaction text

    Returns:
        Configured PromptTemplate
    """
    return SIMPLE_PARSER_PROMPT.format(transaction_line=transaction_line)


# Prompt for extracting insights from parsed data
PARSING_INSIGHT_PROMPT = """
You are a FinanceIQ parsing assistant. Based on the extracted transaction data below,
provide a summary of the parsing results:

Extracted Transactions Summary:
- Total transactions: {total_count}
- Date range: {date_range}
- Total amount extracted: {total_amount}

Parsing observations:
{parsing_notes}

Provide a brief summary of the extraction quality and any notable observations.
"""

def get_parsing_insight_prompt(total_count: int, date_range: str,
                                total_amount: float, parsing_notes: str) -> str:
    """Generate a parsing insight prompt."""
    return PARSING_INSIGHT_PROMPT.format(
        total_count=total_count,
        date_range=date_range,
        total_amount=total_amount,
        parsing_notes=parsing_notes
    )
