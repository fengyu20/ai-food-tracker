from abc import ABC, abstractmethod
from typing import Dict, Any

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