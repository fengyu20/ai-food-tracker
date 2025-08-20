from abc import ABC, abstractmethod
from typing import Dict, Any
import os

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install the 'openai' library: pip install openai")

def get_openrouter_client() -> OpenAI:
    """Initializes and returns the OpenAI client for OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

class BaseProvider(ABC):
    """
    Abstract Base Class.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    @property
    def name(self) -> str:
        """Returns the provider's name from its class name."""
        return self.__class__.__name__.replace("Provider", "").lower()
    
    @abstractmethod
    def classify(
        self,
        image_path: str,
        prompt: str,
        model_name: str,
        temperature: float,
    ) -> Dict[str, Any]:
        """
        Analyzes an image using the specified model and prompt.
        """
        pass