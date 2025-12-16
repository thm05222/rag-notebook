# Apply langchain patch before importing esperanto
# This fixes ModelProfile import errors in esperanto
from open_notebook.utils.langchain_patch import *  # noqa: F401, F403

from typing import Optional

from esperanto import LanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from loguru import logger

from open_notebook.domain.models import model_manager


async def provision_langchain_model(
    content, model_id, default_type, **kwargs
) -> BaseChatModel:
    """
    Get model instance based on configuration.
    
    Args:
        content: Input content (for logging purposes only, no longer used for token-based model selection)
        model_id: Specified model ID. If None, will be retrieved from role default configuration.
        role: Role name (orchestrator, executor, refiner, etc.) - used to look up default model type from config
        **kwargs: Additional model parameters
    
    Returns:
        BaseChatModel instance
    
    Raises:
        ValueError: If model is not configured for the specified role
    """
    # If model_id is not specified, get default model type from role configuration
    if model_id is None:
        defaults = await model_manager.get_defaults()
        role_types = defaults.role_default_types or {}
        default_type = role_types.get(role)
        
        if default_type:
            model_id = getattr(defaults, f"default_{default_type}_model", None)
        
        if not model_id:
            raise ValueError(
                f"Model not configured for role '{role}'. "
                f"Please configure a model in Settings > Models."
            )
    
    # Get model directly, no automatic selection logic
    model = await model_manager.get_model(model_id, **kwargs)
    
    if model is None:
        raise ValueError(
            f"Model with ID '{model_id}' not found for role '{role}'. "
            f"Please check your model configuration."
        )
    
    assert isinstance(model, LanguageModel), f"Model is not a LanguageModel: {model}"
    langchain_model = model.to_langchain()
    
    # Fix parameter conflicts for Anthropic models
    # esperanto library sets default temperature=1.0 and top_p=0.9, which causes conflicts
    if isinstance(langchain_model, ChatAnthropic):
        if langchain_model.temperature is not None and langchain_model.top_p is not None:
            logger.debug(
                f"Removing top_p={langchain_model.top_p} from Anthropic model "
                f"(temperature={langchain_model.temperature} already set)"
            )
            langchain_model = langchain_model.copy(update={'top_p': None})
    
    return langchain_model
