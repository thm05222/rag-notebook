from esperanto import LanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from loguru import logger

from open_notebook.domain.models import model_manager
from open_notebook.utils import token_count


async def provision_langchain_model(
    content, model_id, default_type, **kwargs
) -> BaseChatModel:
    """
    Returns the best model to use based on the context size and on whether there is a specific model being requested in Config.
    If context > 105_000, returns the large_context_model
    If model_id is specified in Config, returns that model
    Otherwise, returns the default model for the given type
    """
    tokens = token_count(content)

    if tokens > 105_000:
        logger.debug(
            f"Using large context model because the content has {tokens} tokens"
        )
        model = await model_manager.get_default_model("large_context", **kwargs)
    elif model_id:
        model = await model_manager.get_model(model_id, **kwargs)
    else:
        model = await model_manager.get_default_model(default_type, **kwargs)

    logger.debug(f"Using model: {model}")
    
    # Check if model is None and provide helpful error message
    if model is None:
        if model_id:
            raise ValueError(
                f"Model with ID '{model_id}' not found. Please check your model configuration."
            )
        elif tokens > 105_000:
            raise ValueError(
                f"No large context model configured. Please set a large_context_model in your default models configuration."
            )
        else:
            model_type_display = {
                "chat": "chat model",
                "transformation": "transformation model",
                "tools": "tools model",
                "embedding": "embedding model",
            }.get(default_type, default_type)
            raise ValueError(
                f"No default {model_type_display} configured. Please set a default_{default_type}_model in your default models configuration."
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
