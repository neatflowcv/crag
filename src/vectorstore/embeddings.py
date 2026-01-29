import logging
import os
import sys
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

from unittest.mock import patch

import tqdm

_original_tqdm = tqdm.tqdm


class SilentTqdm(_original_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


tqdm.tqdm = SilentTqdm
sys.modules["tqdm"].tqdm = SilentTqdm
sys.modules["tqdm.auto"].tqdm = SilentTqdm

import transformers
from sentence_transformers import SentenceTransformer

from src.config.settings import settings

transformers.logging.set_verbosity_error()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model
