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
    """임베딩 모델을 로드합니다. 로컬에 모델이 있으면 그것을 사용하고, 없으면 다운로드 후 저장합니다."""
    global _model
    if _model is None:
        local_path = settings.embedding_model_local_dir

        if local_path.exists() and any(local_path.iterdir()):
            # 로컬에 모델이 있으면 로컬에서 로드
            _model = SentenceTransformer(str(local_path))
        else:
            # 로컬에 모델이 없으면 다운로드 후 저장
            _model = SentenceTransformer(settings.embedding_model)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            _model.save(str(local_path))

    return _model
