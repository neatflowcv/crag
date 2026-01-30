import logging
import warnings

import transformers
from sentence_transformers import SentenceTransformer

from crag.config.settings import settings

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
