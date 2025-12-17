from typing import ClassVar, Dict, Optional, Union
from pydantic import Field

# Apply langchain patch before importing esperanto
# This fixes ModelProfile import errors in esperanto
from open_notebook.utils.langchain_patch import *  # noqa: F401, F403

from esperanto import (
    AIFactory,
    EmbeddingModel,
    LanguageModel,
)
from loguru import logger

from open_notebook.database.repository import repo_query
from open_notebook.domain.base import ObjectModel, RecordModel

ModelType = Union[LanguageModel, EmbeddingModel]


class Model(ObjectModel):
    table_name: ClassVar[str] = "model"
    name: str
    provider: str
    type: str

    @classmethod
    async def get_models_by_type(cls, model_type):
        models = await repo_query(
            "SELECT * FROM model WHERE type=$model_type;", {"model_type": model_type}
        )
        return [Model(**model) for model in models]


class DefaultModels(RecordModel):
    record_id: ClassVar[str] = "open_notebook:default_models"
    default_chat_model: Optional[str] = None
    default_transformation_model: Optional[str] = None
    large_context_model: Optional[str] = None  # Deprecated: no longer used for auto-selection
    # default_vision_model: Optional[str]
    default_embedding_model: Optional[str] = None
    default_tools_model: Optional[str] = None
    
    # Role-specific default model IDs (direct model assignment)
    # Maps role names directly to model IDs - if set, takes precedence over role_default_types
    role_default_models: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {
            "orchestrator": None,
            "executor": None,
            "refiner": None,
        }
    )
    
    # Role-specific default model type mappings
    # Maps role names to default model types (chat, tools, transformation, embedding)
    # Used as fallback if role_default_models is not set (deprecated - will be removed in future)
    role_default_types: Dict[str, str] = Field(
        default_factory=lambda: {
            "orchestrator": "tools",
            "executor": "tools",
            "refiner": "chat",
            "synthesis": "chat",
            "evaluation": "tools",
            "transformation": "transformation",
            "source_chat": "chat",
            "prompt": "transformation",
            "strategy": "tools",
            "answer": "tools",
            "final_answer": "chat",
            "decision": "tools",
            "correction": "tools",
            "fallback": "chat",
        }
    )


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._model_cache: Dict[str, ModelType] = {}
            self._default_models = None

    async def get_model(self, model_id: str, **kwargs) -> Optional[ModelType]:
        if not model_id:
            return None

        cache_key = f"{model_id}:{str(kwargs)}"

        if cache_key in self._model_cache:
            cached_model = self._model_cache[cache_key]
            if not isinstance(
                cached_model,
                (LanguageModel, EmbeddingModel),
            ):
                raise TypeError(
                    f"Cached model is of unexpected type: {type(cached_model)}"
                )
            return cached_model

        try:
            model: Model = await Model.get(model_id)
        except Exception as e:
            # Log the original error for debugging
            logger.debug(f"Failed to get model {model_id}: {str(e)}")
            # Raise a clear error message with the model ID
            raise ValueError(f"Model with ID '{model_id}' not found. Please check your model configuration.")

        if not model.type or model.type not in [
            "language",
            "embedding",
        ]:
            raise ValueError(f"Invalid model type: {model.type}")

        model_instance: ModelType
        if model.type == "language":
            # Fix parameter conflicts for Anthropic models
            # Anthropic API doesn't allow both temperature and top_p to be specified
            config = dict(kwargs)
            if model.provider.lower() == "anthropic":
                if "temperature" in config and "top_p" in config:
                    # Anthropic prefers temperature over top_p when both are present
                    logger.debug(f"Removing top_p parameter for Anthropic model {model.name} (temperature already specified)")
                    config.pop("top_p", None)
                elif "top_p" in config and "temperature" not in config:
                    # If only top_p is specified, that's fine
                    pass
            model_instance = AIFactory.create_language(
                model_name=model.name,
                provider=model.provider,
                config=config,
            )
        elif model.type == "embedding":
            try:
                model_instance = AIFactory.create_embedding(
                    model_name=model.name,
                    provider=model.provider,
                    config=kwargs,
                )
            except Exception as e:
                logger.error(f"Failed to create embedding model instance: {e}")
                logger.exception(e)
                raise ValueError(f"Failed to initialize embedding model '{model.name}' from provider '{model.provider}': {str(e)}")
        else:
            raise ValueError(f"Invalid model type: {model.type}")

        self._model_cache[cache_key] = model_instance
        return model_instance

    def clear_cache(self):
        """Clear all cached model instances"""
        self._model_cache.clear()
        logger.info("Model cache cleared")

    async def clean_invalid_model_references(self, auto_save: bool = True) -> List[str]:
        """
        Clean up invalid model references from default models configuration.
        
        Args:
            auto_save: If True, automatically save cleaned configuration to database.
                      If False, only return the list of cleaned fields without saving.
        
        Returns:
            List of field names that were cleaned (e.g., ["default_chat_model", "role_default_models[orchestrator]"])
        
        Note:
            This method should be called explicitly when needed (e.g., after deleting a model,
            or when updating default models configuration). It should NOT be called automatically
            during read operations to avoid side effects.
        """
        if not self._default_models:
            await self.refresh_defaults()
        
        try:
            all_models = await Model.get_all()
            valid_model_ids = {model.id for model in all_models if model.id}
            cleaned_fields = []
            needs_update = False
            
            # Check and clean each default model field for invalid references
            if self._default_models.default_chat_model and self._default_models.default_chat_model not in valid_model_ids:  # type: ignore[attr-defined]
                logger.warning(f"Cleaning invalid default_chat_model reference: {self._default_models.default_chat_model}")  # type: ignore[attr-defined]
                self._default_models.default_chat_model = None  # type: ignore[attr-defined]
                cleaned_fields.append("default_chat_model")
                needs_update = True
            
            if self._default_models.default_transformation_model and self._default_models.default_transformation_model not in valid_model_ids:  # type: ignore[attr-defined]
                logger.warning(f"Cleaning invalid default_transformation_model reference: {self._default_models.default_transformation_model}")  # type: ignore[attr-defined]
                self._default_models.default_transformation_model = None  # type: ignore[attr-defined]
                cleaned_fields.append("default_transformation_model")
                needs_update = True
            
            if self._default_models.large_context_model and self._default_models.large_context_model not in valid_model_ids:  # type: ignore[attr-defined]
                logger.warning(f"Cleaning invalid large_context_model reference: {self._default_models.large_context_model}")  # type: ignore[attr-defined]
                self._default_models.large_context_model = None  # type: ignore[attr-defined]
                cleaned_fields.append("large_context_model")
                needs_update = True
            
            if self._default_models.default_embedding_model and self._default_models.default_embedding_model not in valid_model_ids:  # type: ignore[attr-defined]
                logger.warning(f"Cleaning invalid default_embedding_model reference: {self._default_models.default_embedding_model}")  # type: ignore[attr-defined]
                self._default_models.default_embedding_model = None  # type: ignore[attr-defined]
                cleaned_fields.append("default_embedding_model")
                needs_update = True
            
            if self._default_models.default_tools_model and self._default_models.default_tools_model not in valid_model_ids:  # type: ignore[attr-defined]
                logger.warning(f"Cleaning invalid default_tools_model reference: {self._default_models.default_tools_model}")  # type: ignore[attr-defined]
                self._default_models.default_tools_model = None  # type: ignore[attr-defined]
                cleaned_fields.append("default_tools_model")
                needs_update = True
            
            # Check and clean role_default_models for invalid references
            role_models = self._default_models.role_default_models or {}  # type: ignore[attr-defined]
            if role_models:
                updated_role_models = {}
                for role, model_id in role_models.items():
                    if model_id and model_id not in valid_model_ids:
                        logger.warning(f"Cleaning invalid role_default_models[{role}] reference: {model_id}")
                        updated_role_models[role] = None
                        cleaned_fields.append(f"role_default_models[{role}]")
                        needs_update = True
                    else:
                        updated_role_models[role] = model_id
                
                if needs_update:
                    self._default_models.role_default_models = updated_role_models  # type: ignore[attr-defined]
            
            # Update database if any fields were cleaned and auto_save is enabled
            if needs_update and auto_save:
                await self._default_models.update()
                logger.info(f"Cleaned {len(cleaned_fields)} invalid model reference(s): {', '.join(cleaned_fields)}")
            elif needs_update:
                logger.info(f"Identified {len(cleaned_fields)} invalid model reference(s) to clean (not saved): {', '.join(cleaned_fields)}")
            
            return cleaned_fields
        except Exception as e:
            # For exceptions, log and return empty list rather than failing
            # This ensures cleanup doesn't break the calling code
            logger.warning(f"Error during model reference cleanup: {e}")
            logger.exception(e)
            return []

    async def refresh_defaults(self):
        """Refresh the default models from the database and clear model cache.
        
        Note: This method only reads from the database and does NOT perform any write operations.
        To clean invalid model references, call clean_invalid_model_references() explicitly.
        """
        self._default_models = await DefaultModels.get_instance()
        
        # Clear the model cache to ensure we use fresh instances with the new defaults
        self.clear_cache()

    async def get_defaults(self) -> DefaultModels:
        """Get the default models configuration (always fetches fresh from DB)"""
        # Always refresh to ensure we have the latest defaults
        # This is important when embedding models are changed
        await self.refresh_defaults()
        if not self._default_models:
            raise RuntimeError("Failed to initialize default models configuration")
        return self._default_models

    async def get_embedding_model(self, **kwargs) -> Optional[EmbeddingModel]:
        """Get the default embedding model"""
        defaults = await self.get_defaults()
        model_id = defaults.default_embedding_model
        if not model_id:
            return None
        model = await self.get_model(model_id, **kwargs)
        assert model is None or isinstance(model, EmbeddingModel), (
            f"Expected EmbeddingModel but got {type(model)}"
        )
        return model

    async def get_default_model(self, model_type: str, **kwargs) -> Optional[ModelType]:
        """
        Get the default model for a specific type.
        
        Note: This method is deprecated for new code. Use provision_langchain_model with role parameter instead.

        Args:
            model_type: The type of model to retrieve (e.g., 'chat', 'embedding', etc.)
            **kwargs: Additional arguments to pass to the model constructor
        
        Returns:
            Model instance or None if not configured
        
        Raises:
            ValueError: If model is not configured (when used via provision_langchain_model)
        """
        defaults = await self.get_defaults()
        model_id = None

        if model_type == "chat":
            model_id = defaults.default_chat_model
        elif model_type == "transformation":
            model_id = defaults.default_transformation_model
        elif model_type == "tools":
            model_id = defaults.default_tools_model
        elif model_type == "embedding":
            model_id = defaults.default_embedding_model
        elif model_type == "large_context":
            model_id = defaults.large_context_model

        if not model_id:
            return None

        try:
            return await self.get_model(model_id, **kwargs)
        except ValueError as e:
            # Provide more specific error message for default model configuration issues
            model_type_display = {
                "chat": "default chat model",
                "transformation": "default transformation model",
                "tools": "default tools model",
                "embedding": "default embedding model",
                "large_context": "large context model",
            }.get(model_type, f"default {model_type} model")
            
            # Log the error for debugging
            logger.warning(
                f"Default model configuration error: {model_type_display} (ID: {model_id}) not found. "
                f"Original error: {str(e)}"
            )
            
            raise ValueError(
                f"The configured {model_type_display} (ID: {model_id}) no longer exists. "
                f"Please update your default model configuration in Settings > Models. "
                f"Original error: {str(e)}"
            )
        except Exception as e:
            # Handle unexpected errors with more context
            model_type_display = {
                "chat": "default chat model",
                "transformation": "default transformation model",
                "tools": "default tools model",
                "embedding": "default embedding model",
                "large_context": "large context model",
            }.get(model_type, f"default {model_type} model")
            
            logger.error(
                f"Unexpected error getting {model_type_display} (ID: {model_id}): {str(e)}"
            )
            
            raise ValueError(
                f"Error loading {model_type_display} (ID: {model_id}): {str(e)}. "
                f"Please check your model configuration in Settings > Models."
            )


model_manager = ModelManager()
