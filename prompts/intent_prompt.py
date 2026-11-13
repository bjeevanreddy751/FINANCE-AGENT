"""
Intent Detection Prompts
Prompt templates for detecting user intent and routing queries.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import List, Optional, Dict

# Available intents
INTENTS = {
    "spending_analysis": "Analyze spending patterns, category breakdown, and expenditure trends",
    "transaction_search": "Find specific transactions, search by description, amount, or date",
    "monthly_comparison": "Compare spending between months, month-over-month analysis",
    "budget_recommendations": "Get budgeting advice, savings suggestions, expense optimization",
    "income_analysis": "Analyze income sources, salary, and earnings",
    "general_financial_questions": "General questions about finances, summary, or other",
}

# Intent descriptions for the LLM
INTENT_DESCRIPTIONS = """
Available Intents and their meanings:

1. spending_analysis: User wants to understand their spending patterns
   - Examples: "How much did I spend?", "Where am I spending most?", "What are my expenses?"
   - Keywords: spend, expense, spending, category, breakdown, pattern

2. transaction_search: User wants to find specific transactions
   - Examples: "Show my Amazon purchases", "Find rent payment", "Highest transaction"
   - Keywords: find, search, show, which, transaction, purchase, payment, highest, lowest

3. monthly_comparison: User wants to compare periods
   - Examples: "Compare last two months", "This month vs last month", "Monthly trend"
   - Keywords: compare, month, months, versus, vs, period, trend, change, difference

4. budget_recommendations: User wants advice on saving and budgeting
   - Examples: "How can I save more?", "Budget suggestions", "Reduce expenses"
   - Keywords: save, savings, budget, recommendation, suggestion, advice, reduce, optimize, improve

5. income_analysis: User wants to understand income
   - Examples: "My total income", "Salary received", "Income sources"
   - Keywords: income, salary, earnings, received, credit, earning

6. general_financial_questions: Other financial questions or general queries
   - Examples: "Help me understand my finances", "Summary", "What can you tell me?"
   - Keywords: help, tell, summary, overview, general, what, information
"""

# System prompt for intent detection
INTENT_SYSTEM_PROMPT = f"""You are a FinanceIQ Intent Detection Agent.
Your task is to classify user queries into one of the predefined intents.

{INTENT_DESCRIPTIONS}

Guidelines:
- Be precise in intent classification
- Consider the overall context, not just keywords
- If multiple intents could apply, choose the primary one
- For ambiguous queries, use "general_financial_questions"
- Return only the intent key (one of: {', '.join(INTENTS.keys())})

Always output in this format:
{{"intent": "intent_key", "confidence": "high|medium|low", "reasoning": "Brief explanation"}}
"""

# Human prompt for intent detection
INTENT_HUMAN_PROMPT = """Classify the intent of this user query:

User Query: "{query}"

User's available data context: {data_context}

Consider what the user is really asking for and classify accordingly.

Return JSON: {{"intent": "...", "confidence": "...", "reasoning": "..."}}
"""

# Create the intent detection chain
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", INTENT_SYSTEM_PROMPT),
    ("human", INTENT_HUMAN_PROMPT),
])

# Alternative: simple keyword-based intent detection
KEYWORD_INTENT_MAP = {
    "spending_analysis": [
        "spend", "expense", "spending", "category", "breakdown", "how much",
        "pattern", "analysis", "expenditure"
    ],
    "transaction_search": [
        "find", "search", "show", "which", "transaction", "purchase",
        "payment", "highest", "lowest", "specific", "particular"
    ],
    "monthly_comparison": [
        "compare", "month", "months", "versus", "vs", "period", "trend",
        "change", "difference", "last month", "this month", "previous"
    ],
    "budget_recommendations": [
        "save", "saving", "savings", "budget", "recommend", "suggest",
        "advice", "reduce", "optimiz", "improve", "cut", "minimize"
    ],
    "income_analysis": [
        "income", "salary", "earning", "received", "credit", "earning",
        "pay", "wage", "bonus", "income source"
    ],
    "general_financial_questions": [
        "help", "tell", "summary", "overview", "general", "what", "information",
        "explain", "understand", "can you"
    ]
}


def detect_intent_with_llm(query: str, data_context: str = "") -> Optional[Dict]:
    """
    Detect user intent using LLM.

    Args:
        query: User's question/query
        data_context: Available data context for the LLM

    Returns:
        Intent dictionary with intent, confidence, and reasoning
    """
    try:
        prompt_value = intent_prompt.format(
            query=query,
            data_context=data_context if data_context else "No data uploaded yet"
        )
        # The actual LLM call happens in the advisor agent
        # This function returns the prompt to be executed
        return {"prompt": prompt_value, "query": query}
    except Exception:
        return None


def detect_intent_keyword_based(query: str) -> Dict:
    """
    Detect intent using keyword matching (fallback method).

    Args:
        query: User's question/query

    Returns:
        Detected intent with confidence
    """
    query_lower = query.lower()

    scores = {}
    for intent, keywords in KEYWORD_INTENT_MAP.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return {
            "intent": "general_financial_questions",
            "confidence": "low",
            "reasoning": "No clear intent keywords detected"
        }

    best_intent, best_score = max(scores.items(), key=lambda x: x[1])

    confidence = "high" if best_score >= 3 else \
                 "medium" if best_score >= 2 else "low"

    return {
        "intent": best_intent,
        "confidence": confidence,
        "reasoning": f"Detected based on keyword matching: {best_score} keyword(s) found"
    }


def route_to_agent(intent: str, query: str, data_context: str) -> str:
    """
    Route the query to the appropriate agent based on intent.

    Args:
        intent: Detected intent
        query: User's original query
        data_context: Available data context

    Returns:
        Routing information for the orchestrator
    """
    routing_map = {
        "spending_analysis": {
            "agent": "analysis_agent",
            "purpose": "Analyze spending patterns and category breakdown",
            "requires_data": True
        },
        "transaction_search": {
            "agent": "analysis_agent",
            "purpose": "Search and retrieve specific transactions",
            "requires_data": True
        },
        "monthly_comparison": {
            "agent": "analysis_agent",
            "purpose": "Compare financial metrics between months",
            "requires_data": True
        },
        "budget_recommendations": {
            "agent": "advisor_agent",
            "purpose": "Generate personalized budgeting recommendations",
            "requires_data": True
        },
        "income_analysis": {
            "agent": "analysis_agent",
            "purpose": "Analyze income patterns and sources",
            "requires_data": True
        },
        "general_financial_questions": {
            "agent": "advisor_agent",
            "purpose": "Provide general financial insights and answers",
            "requires_data": True
        }
    }

    routing = routing_map.get(intent, routing_map["general_financial_questions"])

    # Check if data is available
    if not data_context or "No data uploaded yet" in data_context:
        routing["requires_data"] = False
        routing["reason"] = "No uploaded data available"

    return routing


def get_intent_prompt(query: str, data_summary: str = "") -> ChatPromptTemplate:
    """
    Get a configured intent detection prompt.

    Args:
        query: User's query
        data_summary: Brief summary of available data

    Returns:
        Configured prompt template
    """
    context = data_summary if data_summary else "No financial data has been uploaded yet."
    return intent_prompt.format(query=query, data_context=context)
