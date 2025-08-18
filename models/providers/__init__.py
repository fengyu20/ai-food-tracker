from models.providers.gemini import GeminiProvider
from models.providers.openrouter import OpenRouterProvider
from models.providers.llama import LlamaProvider

def get_provider(provider_name: str, api_key: str = None):
    """
    Factory function to get an instance of a model provider.
    """
    if provider_name == "gemini":
        return GeminiProvider(api_key=api_key)
    elif provider_name == "openrouter":
        return OpenRouterProvider(api_key=api_key)
    elif provider_name == "llama":
        return LlamaProvider()
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")