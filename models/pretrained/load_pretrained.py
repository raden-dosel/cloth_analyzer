# src/models/pretrained/load_pretrained.py
import logging
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel
from typing import Optional, Dict, Union
from pathlib import Path
from ..architecture import ClothingPreferenceModel

logger = logging.getLogger(__name__)

class PretrainedLoader:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: Union[str, Path],
        config: Optional[Dict] = None,
        from_hf_hub: bool = False,
        **kwargs
    ) -> ClothingPreferenceModel:
        """
        Load pretrained model from various sources
        
        Args:
            pretrained_path: Can be either:
                - Hugging Face Hub model ID
                - Local directory path
                - Direct model file path
            config: Custom configuration overrides
            from_hf_hub: Force load from Hugging Face Hub
            kwargs: Additional loading arguments
            
        Returns:
            ClothingPreferenceModel: Initialized model with loaded weights
        """
        # Resolve path type
        if from_hf_hub or ":" in str(pretrained_path):
            return cls._load_from_hf_hub(pretrained_path, config, **kwargs)
            
        if Path(pretrained_path).is_dir():
            return cls._load_from_local_dir(pretrained_path, config, **kwargs)
            
        return cls._load_from_file(pretrained_path, config, **kwargs)

    @classmethod
    def _load_from_hf_hub(
        cls,
        model_id: str,
        config: Optional[Dict],
        **kwargs
    ) -> ClothingPreferenceModel:
        """Load model from Hugging Face Hub"""
        logger.info(f"Loading pretrained model from HF Hub: {model_id}")
        
        # Download config
        try:
            hf_config = AutoConfig.from_pretrained(model_id)
            model_config = cls._convert_hf_config(hf_config, config)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise
            
        # Initialize model
        model = ClothingPreferenceModel(**model_config)
        
        # Download and load weights
        try:
            state_dict = cls._get_hf_state_dict(model_id)
            model = cls._load_state_dict(model, state_dict, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load weights: {str(e)}")
            raise
            
        return model

    @classmethod
    def _load_from_local_dir(
        cls,
        model_dir: Union[str, Path],
        config: Optional[Dict],
        **kwargs
    ) -> ClothingPreferenceModel:
        """Load model from local directory"""
        model_dir = Path(model_dir)
        logger.info(f"Loading pretrained model from local directory: {model_dir}")
        
        # Load config
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        model_config = cls._load_config_file(config_path, config)
        
        # Initialize model
        model = ClothingPreferenceModel(**model_config)
        
        # Load weights
        model_file = model_dir / "pytorch_model.bin"
        if not model_file.exists():
            raise FileNotFoundError(f"Model weights not found: {model_file}")
            
        state_dict = torch.load(model_file, map_location="cpu")
        model = cls._load_state_dict(model, state_dict, **kwargs)
        
        return model

    @classmethod
    def _load_from_file(
        cls,
        model_path: Union[str, Path],
        config: Optional[Dict],
        **kwargs
    ) -> ClothingPreferenceModel:
        """Load from direct model file path"""
        logger.info(f"Loading pretrained model from file: {model_path}")
        
        # Try to load config from parent directory
        model_path = Path(model_path)
        config_path = model_path.parent / "config.json"
        model_config = {}
        
        if config_path.exists():
            model_config = cls._load_config_file(config_path, config)
        elif config is None:
            raise ValueError("Config must be provided when loading from model file")
        else:
            model_config = config
            
        # Initialize model
        model = ClothingPreferenceModel(**model_config)
        
        # Load weights
        state_dict = torch.load(model_path, map_location="cpu")
        model = cls._load_state_dict(model, state_dict, **kwargs)
        
        return model

    @staticmethod
    def _convert_hf_config(
        hf_config: AutoConfig,
        custom_config: Optional[Dict]
    ) -> Dict:
        """Convert Hugging Face config to our format"""
        base_config = {
            "base_model": hf_config.model_type,
            "hidden_size": hf_config.hidden_size,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_categories": 7,  # Default, override from custom_config
            "num_sentiments": 2
        }
        
        if custom_config:
            base_config.update(custom_config)
            
        return base_config

    @staticmethod
    def _load_config_file(
        config_path: Path,
        custom_config: Optional[Dict]
    ) -> Dict:
        """Load config file with custom overrides"""
        with open(config_path) as f:
            config = json.load(f)
            
        if custom_config:
            config.update(custom_config)
            
        return config

    @staticmethod
    def _get_hf_state_dict(model_id: str) -> Dict:
        """Download weights from Hugging Face Hub"""
        weights_path = hf_hub_download(
            repo_id=model_id,
            filename="pytorch_model.bin"
        )
        return torch.load(weights_path, map_location="cpu")

    @staticmethod
    def _load_state_dict(
        model: ClothingPreferenceModel,
        state_dict: Dict,
        strict: bool = True,
        verbose: bool = True
    ) -> ClothingPreferenceModel:
        """Load state dict with error handling"""
        current_state = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in current_state}
        missing = [k for k in current_state.keys() if k not in state_dict]
        unexpected = [k for k in state_dict.keys() if k not in current_state]

        # Load matching parameters
        model.load_state_dict(filtered_dict, strict=False)

        # Initialize missing weights
        if missing:
            if verbose:
                logger.warning(f"Missing keys: {missing}")
                logger.info("Initializing missing weights...")
            model.init_weights()

        # Handle unexpected keys
        if unexpected and verbose:
            logger.warning(f"Unexpected keys: {unexpected}")

        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error loading state dict\nMissing: {missing}\nUnexpected: {unexpected}"
            )

        return model

    @classmethod
    def convert_from_vanilla_bert(
        cls,
        bert_model: Union[str, Path],
        output_dir: Union[str, Path],
        config: Optional[Dict] = None
    ) -> ClothingPreferenceModel:
        """
        Convert standard BERT model to ClothingPreferenceModel format
        
        Args:
            bert_model: Path to vanilla BERT model
            output_dir: Directory to save converted model
            config: Custom configuration overrides
        """
        logger.info(f"Converting vanilla BERT model: {bert_model}")
        
        # Load original model
        orig_model = AutoModel.from_pretrained(bert_model)
        
        # Initialize target model
        model_config = {
            "base_model": orig_model.config.model_type,
            "hidden_size": orig_model.config.hidden_size,
            "num_attention_heads": orig_model.config.num_attention_heads
        }
        if config:
            model_config.update(config)
            
        model = ClothingPreferenceModel(**model_config)
        
        # Copy compatible weights
        state_dict = orig_model.state_dict()
        model.load_state_dict(state_dict, strict=False)
        
        # Save converted model
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        
        return model