"""
Categorizer Agent Prompts
Prompt templates for transaction categorization.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Available categories
CATEGORIES = [
    "Food and Dining",
    "Shopping",
    "Transportation",
    "Groceries",
    "Entertainment",
    "Healthcare",
    "Education",
    "Bills and Utilities",
    "Rent",
    "Travel",
    "Salary and Income",
    "Transfers",
    "Other"
]

# Category descriptions with examples
CATEGORY_GUIDE = """
Category Classification Guide:

1. Food and Dining: Restaurants, cafes, food delivery (Swiggy, Zomato, UberEats), 
   quick service restaurants, dining out

2. Shopping: Online shopping (Amazon, Flipkart, Myntra), retail purchases, 
   electronics, clothing, gadgets

3. Transportation: Uber, Ola, taxi, metro, train, bus, fuel/petrol, parking,
   auto-rickshaw rides

4. Groceries: Supermarkets, grocery stores, vegetable markets, 
   daily essentials, Nature's Basket, local grocery

5. Entertainment: Netflix, Spotify, movies (BookMyShow), gaming, 
   subscriptions, streaming services, OTT platforms

6. Healthcare: Hospital, doctor, pharmacy, medicine, clinic, 
   diagnostic tests, medical expenses

7. Education: Books, courses, tuition, coaching, 
   educational materials, learning platforms

8. Bills and Utilities: Electricity bill, water bill, gas bill, 
   broadband/internet, phone recharges, utility payments

9. Rent: House rent, flat rent, accommodation payments

10. Travel: Flight bookings, hotel stays, vacation, trips,
    MakeMyTrip, hotels, travel agencies

11. Salary and Income: Salary credit, bonus, freelance payment,
    freelance income, payment received, income

12. Transfers: Bank transfers, UPI transfers, money sent,
    money received from others, internal transfers

13. Other: Anything that doesn't fit above categories,
    miscellaneous, general expenses
"""

# System prompt for categorization
CATEGORIZER_SYSTEM_PROMPT = f"""You are a FinanceIQ Transaction Categorizer Agent.
Your task is to classify transactions into appropriate spending categories.

{CATEGORY_GUIDE}

Guidelines:
- Use the rule-based keywords first (they are most reliable)
- If rule-based classification is uncertain, use your understanding of the transaction
- Consider the merchant name, description, and context
- Be consistent with category assignments
- When in doubt, use "Other" category
- Income (credits) should be categorized as "Salary and Income" or "Transfers"
- Expenses should be categorized based on what was purchased

Available categories: {", ".join(CATEGORIES)}
"""

# Human prompt template for categorization
CATEGORIZER_HUMAN_PROMPT = """Please categorize the following transaction:

Transaction:
- Date: {date}
- Description: {description}
- Amount: {amount}
- Current Category: {current_category}

Based on the description and your knowledge:
1. Suggest the most appropriate category
2. Provide confidence level (high/medium/low)
3. Explain your reasoning briefly

Return in JSON format:
{{
  "category": "Category Name",
  "confidence": "high|medium|low",
  "reasoning": "Brief explanation"
}}
"""

# Create the categorization prompt chain
categorizer_prompt_template = ChatPromptTemplate.from_messages([
    ("system", CATEGORIZER_SYSTEM_PROMPT),
    ("human", CATEGORIZER_HUMAN_PROMPT),
])

# Simple prompt for single transaction classification
SINGLE_CATEGORY_PROMPT = PromptTemplate(
    input_variables=["description", "amount", "context"],
    template="""Classify this transaction into one of these categories:

Available Categories: {categories}

Transaction Description: {description}
Amount: {amount}
Additional Context: {context}

Return only the category name. If unclear, return "Other".
""",
    partial_variables={"categories": ", ".join(CATEGORIES)}
)

# Rule-based keyword mapping (used as fallback and hint)
RULE_BASED_KEYWORDS = {
    "Food and Dining": [
        "restaurant", "cafe", "food", "dining", "lunch", "dinner", "breakfast",
        "swiggy", "zomato", "ubereats", "dominos", "mcdonalds", "kfc", "pizza",
        "foodpanda", "eat", "order food", "meal", "snacks", "hotel food",
        "fine dining", "bar", "pub"
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "shopping", "store", "mall", "retail",
        "electronics", "mobile", "phone", "laptop", "gadget", "clothing",
        "fashion", "apparel", "shoes", "accessories", "purchase", "order",
        "online shopping", "cart", "wishlist"
    ],
    "Transportation": [
        "uber", "ola", "taxi", "cab", "metro", "train", "bus", "fuel",
        "petrol", "diesel", "gas", "parking", "autorickshaw", "auto",
        "commute", "ride", "travel local", "transport", "cng", "rickshaw",
        "two wheeler", "bike fuel"
    ],
    "Groceries": [
        "grocery", "supermarket", "groceries", "vegetable", "fruits", "market",
        "dairy", "eggs", "milk", "bread", "grocer", "nature's basket",
        "food essentials", "daily supplies", " Provision", "pantry"
    ],
    "Entertainment": [
        "netflix", "spotify", "prime", "disney", "hotstar", "ott", "streaming",
        "movie", "cinema", "theatre", "bookmyshow", "tickets", "concert",
        "event", "show", "gaming", "game", "subscription entertainment",
        "music", "video", "entertainment app"
    ],
    "Healthcare": [
        "hospital", "doctor", "clinic", "medical", "pharmacy", "medicine",
        "health", "dentist", "diagnostic", "lab", "test", "treatment",
        "surgery", "sick", "illness", "prescription", "health insurance",
        "medical expense", "healthcare"
    ],
    "Education": [
        "book", "books", "course", "education", "tuition", "coaching",
        "class", "training", "learning", "study", "student", "school",
        "college", "university", "exam", "certification", "workshop",
        "educational", "literature", "academic", "training program"
    ],
    "Bills and Utilities": [
        "bill", "bills", "electricity", "power", "electric", "water", "gas",
        "utility", "broadband", "internet", "wifi", "fiber", "phone bill",
        "mobile bill", "recharge", "topup", "utility bill", "domicile",
        "housekeeping", "maintenance bill"
    ],
    "Rent": [
        "rent", "flat", "house rent", "accommodation", "lease",
        "property rent", "rental", "housing"
    ],
    "Travel": [
        "flight", "airline", "airways", "booking", "hotel", "stay",
        "vacation", "trip", "holiday", "travel", "tour", "make my trip",
        "makemytrip", "expedia", "hotels", "resort", "lodging",
        "travel booking", "vacation rental"
    ],
    "Salary and Income": [
        "salary", "income", "payroll", "wage", "earnings", "bonus",
        "commission", "freelance payment", "payment received", "credit",
        "salary credited", "employement income", "professional fee",
        "income from", "earnings from"
    ],
    "Transfers": [
        "transfer", "upi", "neft", "imps", "rtgs", "bank transfer",
        "money transfer", "sent", "received", "person to person",
        "p2p", "wallet transfer", "payment to", "payment from",
        "fund transfer", "internal transfer"
    ]
}


def get_categorizer_prompt(date: str, description: str, amount: float,
                            current_category: str = "Uncategorized") -> ChatPromptTemplate:
    """
    Get a categorization prompt for a specific transaction.

    Args:
        date: Transaction date
        description: Transaction description
        amount: Transaction amount
        current_category: Current/assigned category

    Returns:
        Configured ChatPromptTemplate
    """
    return categorizer_prompt_template.format(
        date=date,
        description=description,
        amount=amount,
        current_category=current_category
    )


def get_category_from_keywords(description: str) -> str:
    """
    Get category suggestion based on keyword matching.

    Args:
        description: Transaction description

    Returns:
        Suggested category or "Other"
    """
    description_lower = description.lower()

    # Pick the longest matching keyword across all categories. Without this,
    # "Phone Recharge" matches "phone" (Shopping) before "recharge" (Bills).
    best_category = "Other"
    best_length = 0

    for category, keywords in RULE_BASED_KEYWORDS.items():
        for keyword in keywords:
            if keyword in description_lower and len(keyword) > best_length:
                best_category = category
                best_length = len(keyword)

    return best_category


def suggest_categories_for_batch(transactions):
    """
    Suggest categories for a batch of transactions using rule-based approach.

    Args:
        transactions: List of transaction dictionaries

    Returns:
        List of transactions with suggested categories
    """
    result = []

    for txn in transactions:
        category = get_category_from_keywords(txn.get('description', ''))

        txn_copy = txn.copy()
        txn_copy['suggested_category'] = category
        txn_copy['category_source'] = 'rule-based'

        result.append(txn_copy)

    return result
