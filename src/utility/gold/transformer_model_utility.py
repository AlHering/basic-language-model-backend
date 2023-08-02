# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, Optional, List
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor
from langchain.llms import LlamaCpp
from src.configuration import configuration as cfg

"""
Loader classes
"""


class LanguageModel(ABC):
    """
    Abstract language model class.
    """

    @abstractmethod
    def generate(prompt: str) -> Any:
        """
        Main handler method for wrapping language model capabilities.
        :param prompt: User prompt.
        :return: Response.
        """
        pass


class LlamaCppLM(LanguageModel):
    """
    General LM class for LlamaCpp.
    """

    def __init__(self, representation: dict) -> None:
        """
        Initiation method.
        :param representation: Language model representation.
        """
        self.llm = LlamaCpp(
            model_path=os.path.join(
                cfg.PATHS.TEXTGENERATION_MODEL_PATH, representation["path"]),
            n_ctx=representation["context"],
            verbose=representation["verbose"]
        )

    def generate(self, prompt: str) -> Optional[Any]:
        """
        Generation method.
        :param prompt: User prompt.
        :return: Response, if generation method is available else None.
        """
        return self.llm.generate([prompt])


class LocalHFLM(LanguageModel):
    """
    General LM class for local Huggingface models.
    """

    def __init__(self, representation: dict) -> None:
        """
        Initiation method.
        :param representation: Language model representation.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=representation["path"],
            local_files_only=True)
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=representation["path"],
            local_files_only=True)

    def generate(self, prompt: str) -> Optional[Any]:
        """
        Generation method.
        :param prompt: User prompt.
        :return: Response, if generation method is available else None.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return self.model(**inputs)


class LocalHFEmbeddingLM(LanguageModel):
    """
    General LM class for local Huggingface models for embedding.
    """

    def __init__(self, representation: dict) -> None:
        """
        Initiation method.
        :param representation: Language model representation.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=representation["path"],
            local_files_only=True)
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=representation["path"],
            local_files_only=True)

    def generate(self, prompt: str) -> List[float]:
        """
        Method for embedding prompt.
        :param prompt: Prompt.
        :return: Prompt embedding.
        """
        inputs = self.tokenizer(prompt, max_length=512,
                                padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**inputs)
        embeddings = self.average_pool(outputs.last_hidden_state,
                                       inputs['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Average pooling function, taken from https://huggingface.co/intfloat/e5-large-v2.
        """
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


"""
Parameter gateways
"""


"""
Parameterized Language Models
"""
SUPPORTED_TYPES = {
    "llamacpp": {
        "loaders": {
            "_default": LlamaCppLM
        },
        "gateways": {}
    },
    "openai": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "gpt4all": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "bedrock": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "cohere": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "google_palm": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "huggingface": {
        "loaders": {
            "_default": LocalHFLM,
            "embedding": LocalHFEmbeddingLM
        },
        "gateways": {}
    },
    "koboldai": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "mosaicml": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "replicate": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "anthropic": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "openllm": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "openlm": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "rwkv": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    }

}


def spawn_language_model_instance(config: str) -> Optional[LanguageModel]:
    """
    Function for spawning language model instance based on configuration.
    :param config: Instance configuration.
    :return: Language model instance if configuration was successful else None.
    """
    lm = SUPPORTED_TYPES.get(config.get("type"), {}).get(
        "loaders", {}).get(config.get("loader", "_default"))
    if lm is not None:
        lm = lm(config)
    return lm
