"""
LLM Client Module
Handles LLM integration for FinanceIQ.
Supports OpenAI, Anthropic, and local models via Ollama.
"""

import os
from typing import Optional, Any
from dataclasses import dataclass

# LangChain imports
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class LLMClient:
    """Client for interacting with various LLM providers."""

    def __init__(self, config: LLMConfig = None):
        self.config = config or self._load_default_config()
        self._llm = None
        self._load_llm()

    def _load_default_config(self) -> LLMConfig:
        """Load configuration from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")

        return LLMConfig(
            provider=provider,
            model=model,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            api_key=api_key,
            base_url=base_url
        )

    def _load_llm(self):
        """Load the appropriate LLM based on configuration."""
        if self.config.provider == "openai":
            self._llm = self._create_openai_llm()
        elif self.config.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self._llm = self._create_anthropic_llm()
        else:
            self._llm = None

    def _create_openai_llm(self):
        """Create OpenAI LLM instance."""
        if not self.config.api_key:
            return None

        kwargs = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "api_key": self.config.api_key,
        }

        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url

        return ChatOpenAI(**kwargs)

    def _create_anthropic_llm(self):
        """Create Anthropic LLM instance."""
        if not self.config.api_key:
            return None

        return ChatAnthropic(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key,
        )

    @property
    def llm(self):
        """Get the LLM instance."""
        return self._llm

    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self._llm is not None

    def invoke(self, messages):
        """Invoke the LLM with messages."""
        if not self._llm:
            raise RuntimeError("LLM not available. Check configuration.")
        return self._llm.invoke(messages)

    def invoke_prompt(self, prompt: str, system_prompt: str = None) -> str:
        """Invoke LLM with a simple prompt."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = self.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)


class MockLLM:
    """Mock LLM for demo mode when no API key is available."""

    def __init__(self):
        self._counter = 0

    def invoke(self, messages):
        """Return contextual mock responses."""
        self._counter += 1
        last_msg = messages[-1].content.lower() if messages else ""

        if any(w in last_msg for w in ['spend', 'expense', 'category', 'where']):
            return AIMessage(content=self._mock_spending_response())
        elif any(w in last_msg for w in ['save', 'budget', 'recommend', 'suggest', 'reduce']):
            return AIMessage(content=self._mock_budget_response())
        elif any(w in last_msg for w in ['income', 'salary', 'earn']):
            return AIMessage(content="Based on your uploaded statements, your total income from credits is available in the dashboard summary.")
        elif any(w in last_msg for w in ['month', 'compare', 'last']):
            return AIMessage(content="Monthly comparison data is available in the Financial Analytics section of the dashboard.")
        else:
            return AIMessage(content=self._mock_general_response())

    def _mock_spending_response(self):
        return """## Spending Analysis

Based on your uploaded financial data:

**Total Expenses:** Check the Dashboard for your exact total expenses.

**Highest Spending Category:** Your highest spending category is displayed in the analytics section. Review the Category Breakdown chart for details.

**Key Observations:**
- Track your spending patterns in the Transactions view
- Review category-wise breakdown in Financial Analytics
- Look for trends in the monthly charts

For specific numbers, please refer to your Dashboard and Analytics pages where all calculations are performed using your actual transaction data."""

    def _mock_budget_response(self):
        return """## Budget Recommendations

Based on general financial principles:

1. **Track Your Spending:** Use the Transactions page to monitor where your money goes.

2. **Set Category Budgets:** Identify your top spending categories and set monthly limits.

3. **Review Subscriptions:** Check for recurring payments you may not need.

4. **Build Emergency Fund:** Aim to save 3-6 months of expenses.

5. **Monthly Review:** Compare your spending month-over-month in the Analytics section.

*Note: These are general suggestions. For personalized advice, analyze your specific transaction patterns in the dashboard.*"""

    def _mock_general_response(self):
        return """I'd be happy to help you understand your finances!

I can assist with:
- **Spending Analysis:** "How much did I spend?"
- **Category Questions:** "Where am I spending the most?"
- **Monthly Comparisons:** "Compare my last two months"
- **Budget Tips:** "How can I save more?"

Please upload your bank statements first, then ask me anything about your finances! For exact figures, check the Dashboard and Analytics sections."""

    def __call__(self, messages):
        return self.invoke(messages)


def get_llm_client() -> LLMClient:
    """Get a configured LLM client."""
    return LLMClient()


def get_llm_or_mock() -> Any:
    """Get LLM if available, otherwise return mock for demo mode."""
    client = LLMClient()
    if client.is_available():
        return client.llm
    return MockLLM()
