"""
Patch for langchain_core.language_models to support ModelProfile import.

This patch fixes the ImportError when esperanto tries to import ModelProfile
from langchain_core.language_models. ModelProfile may have been moved to
langchain_core.language_models.profile or langchain_model_profiles.
"""

import langchain_core.language_models as _lm_module

# Check if ModelProfile already exists
if not hasattr(_lm_module, "ModelProfile"):
    # Try to import ModelProfile from the new location and add it to langchain_core.language_models
    _patched = False
    
    # First, try importing from langchain_core.language_models.profile
    try:
        from langchain_core.language_models.profile import ModelProfile, ModelProfileRegistry  # noqa: F401
        _lm_module.ModelProfile = ModelProfile
        _lm_module.ModelProfileRegistry = ModelProfileRegistry
        _patched = True
    except (ImportError, AttributeError):
        pass
    
    # If that doesn't work, try langchain_model_profiles
    if not _patched:
        try:
            from langchain_model_profiles import ModelProfile, ModelProfileRegistry  # noqa: F401
            _lm_module.ModelProfile = ModelProfile
            _lm_module.ModelProfileRegistry = ModelProfileRegistry
            _patched = True
        except ImportError:
            pass
    
    # If all else fails, create a dummy class to prevent import errors
    # This is a fallback to prevent esperanto from failing
    if not _patched:
        class ModelProfile:  # noqa: N801
            """Dummy ModelProfile class to prevent import errors."""
            pass
        
        class ModelProfileRegistry:  # noqa: N801
            """Dummy ModelProfileRegistry class to prevent import errors."""
            pass
        
        _lm_module.ModelProfile = ModelProfile
        _lm_module.ModelProfileRegistry = ModelProfileRegistry

