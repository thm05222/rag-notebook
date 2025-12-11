import os
from typing import List, Optional

from esperanto import AIFactory
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from api.models import (
    DefaultModelsResponse,
    ModelCreate,
    ModelResponse,
    ProviderAvailabilityResponse,
)
from open_notebook.domain.models import DefaultModels, Model
from open_notebook.exceptions import InvalidInputError

router = APIRouter()


def _check_openai_compatible_support(mode: str) -> bool:
    """
    Check if OpenAI-compatible provider is available for a specific mode.

    Args:
        mode: One of 'LLM', 'EMBEDDING'

    Returns:
        bool: True if either generic or mode-specific env var is set
    """
    generic = os.environ.get("OPENAI_COMPATIBLE_BASE_URL") is not None
    specific = os.environ.get(f"OPENAI_COMPATIBLE_BASE_URL_{mode}") is not None
    return generic or specific


@router.get("/models", response_model=List[ModelResponse])
async def get_models(
    type: Optional[str] = Query(None, description="Filter by model type")
):
    """Get all configured models with optional type filtering."""
    try:
        if type:
            models = await Model.get_models_by_type(type)
        else:
            models = await Model.get_all()
        
        return [
            ModelResponse(
                id=model.id,
                name=model.name,
                provider=model.provider,
                type=model.type,
                created=str(model.created),
                updated=str(model.updated),
            )
            for model in models
        ]
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")


@router.post("/models", response_model=ModelResponse)
async def create_model(model_data: ModelCreate):
    """Create a new model configuration."""
    try:
        # Validate model type
        valid_types = ["language", "embedding"]
        if model_data.type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type. Must be one of: {valid_types}"
            )

        # Check for duplicate model name under the same provider (case-insensitive)
        from open_notebook.database.repository import repo_query
        existing = await repo_query(
            "SELECT * FROM model WHERE string::lowercase(provider) = $provider AND string::lowercase(name) = $name LIMIT 1",
            {"provider": model_data.provider.lower(), "name": model_data.name.lower()}
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_data.name}' already exists for provider '{model_data.provider}'"
            )

        new_model = Model(
            name=model_data.name,
            provider=model_data.provider,
            type=model_data.type,
        )
        await new_model.save()

        return ModelResponse(
            id=new_model.id or "",
            name=new_model.name,
            provider=new_model.provider,
            type=new_model.type,
            created=str(new_model.created),
            updated=str(new_model.updated),
        )
    except HTTPException:
        raise
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating model: {str(e)}")


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model configuration and clean up any references in default models."""
    try:
        model = await Model.get(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Clean up default models references before deleting
        defaults = await DefaultModels.get_instance()
        cleaned_fields = []
        
        # Check and clean each default model field
        if defaults.default_chat_model == model_id:  # type: ignore[attr-defined]
            defaults.default_chat_model = None  # type: ignore[attr-defined]
            cleaned_fields.append("default_chat_model")
        
        if defaults.default_transformation_model == model_id:  # type: ignore[attr-defined]
            defaults.default_transformation_model = None  # type: ignore[attr-defined]
            cleaned_fields.append("default_transformation_model")
        
        if defaults.large_context_model == model_id:  # type: ignore[attr-defined]
            defaults.large_context_model = None  # type: ignore[attr-defined]
            cleaned_fields.append("large_context_model")
        
        if defaults.default_embedding_model == model_id:  # type: ignore[attr-defined]
            defaults.default_embedding_model = None  # type: ignore[attr-defined]
            cleaned_fields.append("default_embedding_model")
        
        if defaults.default_tools_model == model_id:  # type: ignore[attr-defined]
            defaults.default_tools_model = None  # type: ignore[attr-defined]
            cleaned_fields.append("default_tools_model")
        
        # Update defaults if any fields were cleaned
        if cleaned_fields:
            await defaults.update()
            logger.info(
                f"Cleaned up default model references for deleted model {model_id}: {', '.join(cleaned_fields)}"
            )
            
            # Refresh the model manager cache
            from open_notebook.domain.models import model_manager
            await model_manager.refresh_defaults()
        
        # Delete the model
        await model.delete()
        
        return {"message": "Model deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")


@router.get("/models/defaults", response_model=DefaultModelsResponse)
async def get_default_models():
    """Get default model assignments. Automatically cleans up invalid model references."""
    try:
        defaults = await DefaultModels.get_instance()
        
        # Automatically clean up invalid model references
        all_models = await Model.get_all()
        valid_model_ids = {model.id for model in all_models if model.id}
        needs_update = False
        
        # Check and clean each default model field
        if defaults.default_chat_model and defaults.default_chat_model not in valid_model_ids:  # type: ignore[attr-defined]
            logger.warning(f"Cleaning up invalid default_chat_model reference: {defaults.default_chat_model}")  # type: ignore[attr-defined]
            defaults.default_chat_model = None  # type: ignore[attr-defined]
            needs_update = True
        
        if defaults.default_transformation_model and defaults.default_transformation_model not in valid_model_ids:  # type: ignore[attr-defined]
            logger.warning(f"Cleaning up invalid default_transformation_model reference: {defaults.default_transformation_model}")  # type: ignore[attr-defined]
            defaults.default_transformation_model = None  # type: ignore[attr-defined]
            needs_update = True
        
        if defaults.large_context_model and defaults.large_context_model not in valid_model_ids:  # type: ignore[attr-defined]
            logger.warning(f"Cleaning up invalid large_context_model reference: {defaults.large_context_model}")  # type: ignore[attr-defined]
            defaults.large_context_model = None  # type: ignore[attr-defined]
            needs_update = True
        
        if defaults.default_embedding_model and defaults.default_embedding_model not in valid_model_ids:  # type: ignore[attr-defined]
            logger.warning(f"Cleaning up invalid default_embedding_model reference: {defaults.default_embedding_model}")  # type: ignore[attr-defined]
            defaults.default_embedding_model = None  # type: ignore[attr-defined]
            needs_update = True
        
        if defaults.default_tools_model and defaults.default_tools_model not in valid_model_ids:  # type: ignore[attr-defined]
            logger.warning(f"Cleaning up invalid default_tools_model reference: {defaults.default_tools_model}")  # type: ignore[attr-defined]
            defaults.default_tools_model = None  # type: ignore[attr-defined]
            needs_update = True
        
        # Update defaults if any fields were cleaned
        if needs_update:
            await defaults.update()
            logger.info("Automatically cleaned up invalid model references")
            # Refresh the model manager cache
            from open_notebook.domain.models import model_manager
            await model_manager.refresh_defaults()

        return DefaultModelsResponse(
            default_chat_model=defaults.default_chat_model,  # type: ignore[attr-defined]
            default_transformation_model=defaults.default_transformation_model,  # type: ignore[attr-defined]
            large_context_model=defaults.large_context_model,  # type: ignore[attr-defined]
            default_embedding_model=defaults.default_embedding_model,  # type: ignore[attr-defined]
            default_tools_model=defaults.default_tools_model,  # type: ignore[attr-defined]
        )
    except Exception as e:
        logger.error(f"Error fetching default models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching default models: {str(e)}")


@router.put("/models/defaults", response_model=DefaultModelsResponse)
async def update_default_models(defaults_data: DefaultModelsResponse):
    """Update default model assignments. Automatically cleans up invalid references before updating."""
    try:
        defaults = await DefaultModels.get_instance()
        
        # First, clean up all invalid model references before updating
        # This ensures we don't leave invalid references even if user only updates some fields
        all_models = await Model.get_all()
        valid_model_ids = {model.id for model in all_models if model.id}
        cleaned_fields = []
        needs_update = False
        
        # Check and clean each default model field for invalid references
        if defaults.default_chat_model and defaults.default_chat_model not in valid_model_ids:  # type: ignore[attr-defined]
            logger.warning(f"Cleaning up invalid default_chat_model reference: {defaults.default_chat_model}")  # type: ignore[attr-defined]
            defaults.default_chat_model = None  # type: ignore[attr-defined]
            cleaned_fields.append("default_chat_model")
            needs_update = True
        
        if defaults.default_transformation_model and defaults.default_transformation_model not in valid_model_ids:  # type: ignore[attr-defined]
            logger.warning(f"Cleaning up invalid default_transformation_model reference: {defaults.default_transformation_model}")  # type: ignore[attr-defined]
            defaults.default_transformation_model = None  # type: ignore[attr-defined]
            cleaned_fields.append("default_transformation_model")
            needs_update = True
        
        if defaults.large_context_model and defaults.large_context_model not in valid_model_ids:  # type: ignore[attr-defined]
            logger.warning(f"Cleaning up invalid large_context_model reference: {defaults.large_context_model}")  # type: ignore[attr-defined]
            defaults.large_context_model = None  # type: ignore[attr-defined]
            cleaned_fields.append("large_context_model")
            needs_update = True
        
        if defaults.default_embedding_model and defaults.default_embedding_model not in valid_model_ids:  # type: ignore[attr-defined]
            logger.warning(f"Cleaning up invalid default_embedding_model reference: {defaults.default_embedding_model}")  # type: ignore[attr-defined]
            defaults.default_embedding_model = None  # type: ignore[attr-defined]
            cleaned_fields.append("default_embedding_model")
            needs_update = True
        
        if defaults.default_tools_model and defaults.default_tools_model not in valid_model_ids:  # type: ignore[attr-defined]
            logger.warning(f"Cleaning up invalid default_tools_model reference: {defaults.default_tools_model}")  # type: ignore[attr-defined]
            defaults.default_tools_model = None  # type: ignore[attr-defined]
            cleaned_fields.append("default_tools_model")
            needs_update = True
        
        if cleaned_fields:
            logger.info(f"Cleaned up {len(cleaned_fields)} invalid model reference(s) before update: {', '.join(cleaned_fields)}")
        
        # Now update only provided fields (after cleaning invalid references)
        if defaults_data.default_chat_model is not None:
            defaults.default_chat_model = defaults_data.default_chat_model  # type: ignore[attr-defined]
        if defaults_data.default_transformation_model is not None:
            defaults.default_transformation_model = defaults_data.default_transformation_model  # type: ignore[attr-defined]
        if defaults_data.large_context_model is not None:
            defaults.large_context_model = defaults_data.large_context_model  # type: ignore[attr-defined]
        if defaults_data.default_embedding_model is not None:
            defaults.default_embedding_model = defaults_data.default_embedding_model  # type: ignore[attr-defined]
        if defaults_data.default_tools_model is not None:
            defaults.default_tools_model = defaults_data.default_tools_model  # type: ignore[attr-defined]
        
        # Update if we cleaned invalid references or if user provided updates
        if needs_update or (
            defaults_data.default_chat_model is not None
            or defaults_data.default_transformation_model is not None
            or defaults_data.large_context_model is not None
            or defaults_data.default_embedding_model is not None
            or defaults_data.default_tools_model is not None
        ):
            await defaults.update()
            
            # Refresh the model manager cache
            from open_notebook.domain.models import model_manager
            await model_manager.refresh_defaults()
        
        return DefaultModelsResponse(
            default_chat_model=defaults.default_chat_model,  # type: ignore[attr-defined]
            default_transformation_model=defaults.default_transformation_model,  # type: ignore[attr-defined]
            large_context_model=defaults.large_context_model,  # type: ignore[attr-defined]
            default_embedding_model=defaults.default_embedding_model,  # type: ignore[attr-defined]
            default_tools_model=defaults.default_tools_model,  # type: ignore[attr-defined]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating default models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating default models: {str(e)}")


@router.post("/models/defaults/cleanup", response_model=DefaultModelsResponse)
async def cleanup_invalid_model_references():
    """Clean up invalid model references in default models configuration."""
    try:
        defaults = await DefaultModels.get_instance()
        cleaned_fields = []
        
        # Get all existing model IDs to validate references
        all_models = await Model.get_all()
        valid_model_ids = {model.id for model in all_models if model.id}
        
        # Check and clean each default model field
        field_mappings = [
            ("default_chat_model", "default_chat_model"),
            ("default_transformation_model", "default_transformation_model"),
            ("large_context_model", "large_context_model"),
            ("default_embedding_model", "default_embedding_model"),
            ("default_tools_model", "default_tools_model"),
        ]
        
        for attr_name, display_name in field_mappings:
            model_id = getattr(defaults, attr_name, None)  # type: ignore[attr-defined]
            if model_id and model_id not in valid_model_ids:
                setattr(defaults, attr_name, None)  # type: ignore[attr-defined]
                cleaned_fields.append(f"{display_name} (ID: {model_id})")
                logger.info(f"Cleaned up invalid model reference: {display_name} = {model_id}")
        
        # Update defaults if any fields were cleaned
        if cleaned_fields:
            await defaults.update()
            logger.info(
                f"Cleaned up {len(cleaned_fields)} invalid model reference(s): {', '.join(cleaned_fields)}"
            )
            
            # Refresh the model manager cache
            from open_notebook.domain.models import model_manager
            await model_manager.refresh_defaults()
        else:
            logger.info("No invalid model references found to clean up")
        
        return DefaultModelsResponse(
            default_chat_model=defaults.default_chat_model,  # type: ignore[attr-defined]
            default_transformation_model=defaults.default_transformation_model,  # type: ignore[attr-defined]
            large_context_model=defaults.large_context_model,  # type: ignore[attr-defined]
            default_embedding_model=defaults.default_embedding_model,  # type: ignore[attr-defined]
            default_tools_model=defaults.default_tools_model,  # type: ignore[attr-defined]
        )
    except Exception as e:
        logger.error(f"Error cleaning up invalid model references: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error cleaning up invalid model references: {str(e)}"
        )


@router.get("/models/providers", response_model=ProviderAvailabilityResponse)
async def get_provider_availability():
    """Get provider availability based on environment variables."""
    try:
        # Check which providers have API keys configured
        provider_status = {
            "ollama": os.environ.get("OLLAMA_API_BASE") is not None,
            "openai": os.environ.get("OPENAI_API_KEY") is not None,
            "groq": os.environ.get("GROQ_API_KEY") is not None,
            "xai": os.environ.get("XAI_API_KEY") is not None,
            "vertex": (
                os.environ.get("VERTEX_PROJECT") is not None
                and os.environ.get("VERTEX_LOCATION") is not None
                and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None
            ),
            "google": (
                os.environ.get("GOOGLE_API_KEY") is not None
                or os.environ.get("GEMINI_API_KEY") is not None
            ),
            "openrouter": os.environ.get("OPENROUTER_API_KEY") is not None,
            "anthropic": os.environ.get("ANTHROPIC_API_KEY") is not None,
            "elevenlabs": os.environ.get("ELEVENLABS_API_KEY") is not None,
            "voyage": os.environ.get("VOYAGE_API_KEY") is not None,
            "azure": (
                os.environ.get("AZURE_OPENAI_API_KEY") is not None
                and os.environ.get("AZURE_OPENAI_ENDPOINT") is not None
                and os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") is not None
                and os.environ.get("AZURE_OPENAI_API_VERSION") is not None
            ),
            "mistral": os.environ.get("MISTRAL_API_KEY") is not None,
            "deepseek": os.environ.get("DEEPSEEK_API_KEY") is not None,
            "openai-compatible": (
                _check_openai_compatible_support("LLM")
                or _check_openai_compatible_support("EMBEDDING")
            ),
        }
        
        available_providers = [k for k, v in provider_status.items() if v]
        unavailable_providers = [k for k, v in provider_status.items() if not v]

        # Get supported model types from Esperanto
        esperanto_available = AIFactory.get_available_providers()

        # Build supported types mapping only for available providers
        supported_types: dict[str, list[str]] = {}
        for provider in available_providers:
            supported_types[provider] = []

            # Special handling for openai-compatible to check mode-specific availability
            if provider == "openai-compatible":
                # Map Esperanto model types to our environment variable modes
                mode_mapping = {
                    "language": "LLM",
                    "embedding": "EMBEDDING",
                }
                for model_type, mode in mode_mapping.items():
                    if model_type in esperanto_available and provider in esperanto_available[model_type]:
                        if _check_openai_compatible_support(mode):
                            supported_types[provider].append(model_type)
            else:
                # Standard provider detection
                for model_type, providers in esperanto_available.items():
                    if provider in providers:
                        supported_types[provider].append(model_type)
        
        return ProviderAvailabilityResponse(
            available=available_providers,
            unavailable=unavailable_providers,
            supported_types=supported_types
        )
    except Exception as e:
        logger.error(f"Error checking provider availability: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking provider availability: {str(e)}")