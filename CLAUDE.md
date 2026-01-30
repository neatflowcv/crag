# 코딩 규칙

## 패키지 관리

- uv를 사용하여 의존성 관리
- `uv sync`로 의존성 설치
- `uv run`으로 스크립트 실행
- `uvx ruff`로 린트 실행

## 프로젝트 구조

- flat layout 사용 (`crag/` 패키지)
- `__init__.py` 사용 금지
- CLI 진입점: `crag/cli.py`

## 코드 스타일

- ruff로 린트
- 타입 힌트 사용
- 한국어 docstring 허용

## 실행 방법

```bash
uv run crag ingest       # 문서 인제스트
uv run crag run "질문"   # 질문에 답변
```
