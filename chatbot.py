"""
chatbot.py - RAG 챗봇 백엔드

BM25 기반 검색과 Ollama LLM을 결합한 RAG(Retrieval-Augmented Generation) 챗봇 모듈.
- 외부 라이브러리 없이 순수 구현한 BM25 검색 엔진
- CSV 테이블 및 GitLab 이슈를 청크 단위로 인덱싱
- RAG 모드 (데이터 참조) / 자유 대화 모드 지원
- 시간 기반 이슈 조회 (KST 변환 포함)
"""

import os
import re
import math
from collections import Counter
from datetime import datetime

import pandas as pd

import ollama as _ollama
from search import _gitlab_config

# instruction following 및 한국어 지원 우수 모델 우선순위
CHAT_MODEL_PRIORITY = [
    "exaone3.5",     # LG AI 한국어 특화, instruction following 최상
    "bllossom",      # 한국어 Llama-3 파인튜닝
    "gemma2",        # 한국어 우수, instruction following 우수
    "llama3.1",      # 한국어 양호
    "llama3-open-ko",
    "llama-ko", "llama3-ko", "ko-llama",
    "exaone", "solar", "EEVE",
]


def list_installed_models() -> list[str]:
    """Ollama에 설치된 모델 목록 반환."""
    try:
        return [m.model for m in _ollama.list().models]
    except Exception:
        return []


def get_chat_model() -> str:
    """instruction following 우수 + 한국어 지원 모델 우선 선택."""
    installed = list_installed_models()
    for preferred in CHAT_MODEL_PRIORITY:
        for m in installed:
            if preferred.lower() in m.lower():
                return m
    from search import get_available_model
    return get_available_model()


_STOP_TOKENS = [
    "\nuser\n", "\nassistant\n", "\n사용자\n", "\nAI\n",
    "<|user|>", "<|assistant|>", "<|im_start|>", "<|im_end|>",
    "<|eot_id|>", "[INST]", "[/INST]",
]

def _clean_response(text: str) -> str:
    """모델이 역할 토큰 / 가짜 대화를 생성한 경우 해당 지점에서 잘라냄."""
    cut_patterns = [
        r'\n\[질문\]',
        r'\nassistant\b',
        r'\nuser\b',
        r'\n사용자:',
        r'\nAI:',
        r'<\|user\|>',
        r'<\|assistant\|>',
        r'<\|im_start\|>',
        r'</p>',
        r'answer\n',
        r'질문:.*\nassistant',
    ]
    earliest = len(text)
    for pattern in cut_patterns:
        m = re.search(pattern, text)
        if m and m.start() > 0:
            earliest = min(earliest, m.start())
    return text[:earliest].strip()


def _call_messages(messages: list[dict], model: str | None = None) -> str:
    """Ollama multi-turn chat API 호출 (stop 토큰 + 역할 토큰 후처리)."""
    if model is None:
        from search import get_available_model
        model = get_available_model()
    try:
        resp = _ollama.chat(
            model=model,
            messages=messages,
            options={"stop": _STOP_TOKENS, "num_predict": 600},
        )
        text = _clean_response(resp.message.content)
        if not text:
            resp2 = _ollama.chat(model=model, messages=messages,
                                 options={"stop": _STOP_TOKENS, "num_predict": 600})
            text = _clean_response(resp2.message.content)
        return text if text else "응답을 생성하지 못했습니다. 다시 시도해주세요."
    except Exception as e:
        return f"[Ollama 오류] {e}"


# ──────────────────────────────────────────────
# BM25 (외부 라이브러리 없이 순수 구현)
# ──────────────────────────────────────────────

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.docs: list[list[str]] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r'[가-힣]+|[a-z0-9]+', text)
        return tokens

    def index(self, texts: list[str]):
        self.docs = [self._tokenize(t) for t in texts]
        n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(n, 1)
        df: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.idf = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        if not self.docs:
            return []
        qtokens = self._tokenize(query)
        scores: list[float] = []
        for doc in self.docs:
            tf_map = Counter(doc)
            dl = len(doc)
            score = 0.0
            for term in qtokens:
                if term not in self.idf:
                    continue
                tf = tf_map.get(term, 0)
                score += self.idf[term] * (
                    tf * (self.k1 + 1) /
                    (tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
                )
            scores.append(score)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]


# ──────────────────────────────────────────────
# 문서 청크
# ──────────────────────────────────────────────

class Chunk:
    __slots__ = ("text", "source")

    def __init__(self, text: str, source: str):
        self.text   = text
        self.source = source


# ──────────────────────────────────────────────
# RAG 인덱스
# ──────────────────────────────────────────────

class RAGIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._bm25  = BM25()
        self._built = False

    def clear(self):
        self.chunks.clear()
        self._built = False

    def clear_gitlab(self):
        """GitLab 청크만 제거 (테이블 청크 유지)."""
        self.chunks = [c for c in self.chunks if not c.source.startswith("GitLab:")]
        self._built = False

    def add_tables(self, tables: dict[str, pd.DataFrame], max_rows_per_table: int = 200):
        """CSV 테이블 → 행 단위 청크."""
        for tname, df in tables.items():
            for _, row in df.head(max_rows_per_table).iterrows():
                parts = [f"{col}={val}" for col, val in row.items() if pd.notna(val) and str(val).strip()]
                text = f"[테이블:{tname}] " + " | ".join(parts)
                self.chunks.append(Chunk(text, f"테이블:{tname}"))

    def add_gitlab_issues(self, issues: list[dict]):
        """GitLab 이슈 목록 → 청크."""
        for issue in issues:
            iid        = issue.get("iid", "?")
            title      = issue.get("title", "")
            desc       = issue.get("description", "") or ""
            state      = issue.get("state", "")
            labels     = issue.get("labels", [])
            created_at = issue.get("created_at", "")
            updated_at = issue.get("updated_at", "")
            comments   = issue.get("comments", [])
            comment_text = "\n".join(
                f"  [{c.get('author','')}]: {c.get('body','')[:200]}" for c in comments[:5]
            )
            meta_parts = []
            if created_at:
                meta_parts.append(f"등록일시: {created_at}")
            if updated_at:
                meta_parts.append(f"수정일시: {updated_at}")
            if state:
                meta_parts.append(f"상태: {state}")
            if labels:
                meta_parts.append(f"레이블: {', '.join(labels)}")
            text = f"[GitLab #{iid}] {title}"
            if meta_parts:
                text += "\n" + " | ".join(meta_parts)
            text += f"\n{desc[:600]}"
            if comment_text:
                text += f"\n코멘트:\n{comment_text}"
            self.chunks.append(Chunk(text, f"GitLab:#{iid}"))

    def build(self):
        self._bm25.index([c.text for c in self.chunks])
        self._built = True

    def search(self, query: str, k: int = 6) -> list[Chunk]:
        if not self._built:
            self.build()
        results = self._bm25.search(query, k)
        return [self.chunks[i] for i, _ in results]

    @property
    def size(self) -> int:
        return len(self.chunks)


# ──────────────────────────────────────────────
# GitLab 이슈 다건 조회
# ──────────────────────────────────────────────

def fetch_gitlab_issues_bulk(max_issues: int = 100) -> list[dict]:
    """최근 GitLab 이슈 N개를 일괄 조회 (코멘트 포함)."""
    import requests

    gitlab_url, token, project = _gitlab_config()
    if not token or not project:
        return []

    headers  = {"PRIVATE-TOKEN": token}
    encoded  = requests.utils.quote(project, safe="")
    url      = f"{gitlab_url}/api/v4/projects/{encoded}/issues"
    params   = {"per_page": min(max_issues, 100), "order_by": "updated_at", "sort": "desc", "state": "all"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15, verify=False)
        if resp.status_code != 200:
            return []
        issues_raw = resp.json()
    except Exception:
        return []

    issues = []
    base_api = f"{gitlab_url}/api/v4/projects/{encoded}/issues"
    for raw in issues_raw[:max_issues]:
        iid   = raw.get("iid")
        comments = []
        try:
            nr = requests.get(f"{base_api}/{iid}/notes?per_page=20&sort=asc",
                              headers=headers, timeout=8, verify=False)
            if nr.status_code == 200:
                for note in nr.json():
                    if not note.get("system", False):
                        comments.append({"author": note["author"]["name"], "body": note["body"][:300]})
        except Exception:
            pass
        issues.append({
            "iid":         iid,
            "title":       raw.get("title", ""),
            "description": raw.get("description", "") or "",
            "labels":      raw.get("labels", []),
            "state":       raw.get("state", ""),
            "created_at":  raw.get("created_at", ""),
            "updated_at":  raw.get("updated_at", ""),
            "comments":    comments,
        })
    return issues


# ──────────────────────────────────────────────
# 대화 히스토리
# ──────────────────────────────────────────────

class ChatHistory:
    def __init__(self):
        self.messages: list[dict] = []

    def add(self, role: str, content: str, sources: list[str] | None = None):
        self.messages.append({
            "role":    role,
            "content": content,
            "sources": sources or [],
        })

    def clear(self):
        self.messages = []

    def recent_ollama_messages(self, n: int = 6) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages[-n:]]


# ──────────────────────────────────────────────
# 답변 생성
# ──────────────────────────────────────────────

_TIME_KEYWORDS = re.compile(
    r'(\d+\s*분\s*(전|내|이내|안에)'
    r'|\d+\s*시간\s*(전|내|이내|안에)?'
    r'|한\s*시간\s*(전|내|이내|안에)?'
    r'|두\s*시간\s*(전|내|이내|안에)?'
    r'|반\s*시간\s*(전|내|이내|안에)?'
    r'|\d+\s*일\s*(전|내|이내|안에)'
    r'|오늘|최근|방금|금일|어제|이번\s*주)'
)


def _is_time_query(query: str) -> bool:
    return bool(_TIME_KEYWORDS.search(query))


def _extract_created_at(chunk) -> str:
    m = re.search(r'등록일시:\s*(\S+)', chunk.text)
    return m.group(1) if m else ""


def rag_chat(query: str, index: RAGIndex, history: ChatHistory,
             model: str | None = None) -> tuple[str, list[str]]:
    """RAG 모드: BM25 검색 → context 주입 → LLM 답변."""
    if _is_time_query(query):
        # 시간 기반 쿼리: GitLab 청크를 등록일시 최신순으로 정렬
        gitlab_chunks = [c for c in index.chunks if c.source.startswith("GitLab:")]
        chunks = sorted(gitlab_chunks, key=_extract_created_at, reverse=True)[:25]
        if not chunks:
            chunks = index.search(query, k=20)
    else:
        chunks = index.search(query, k=6)

    sources = list(dict.fromkeys(c.source for c in chunks))
    context = "\n\n".join(f"[출처: {c.source}]\n{c.text[:500]}" for c in chunks)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system = (
        f"당신은 게임 데이터 QA 전문가 어시스턴트입니다.\n"
        f"현재 시각: {now_str} (KST)\n"
        "규칙:\n"
        "1. 반드시 아래 [참고 데이터]에 있는 내용만 사용하여 한국어로 답변하세요.\n"
        "2. [참고 데이터]에 없는 이슈 번호·제목·설명을 절대 생성하거나 추측하지 마세요. "
        "이슈 내용은 [참고 데이터]에서 복사·인용만 하세요.\n"
        "3. 출처(테이블명 또는 GitLab 이슈 번호)를 반드시 언급하세요.\n"
        "4. 시간 관련 질문 시: [참고 데이터]의 '등록일시'(UTC)를 KST(+9시간)로 변환 후 "
        "현재 시각과 비교하여 조건에 맞는 이슈만 나열하세요. "
        "조건에 맞는 이슈가 없으면 '해당 시간대에 등록된 이슈가 없습니다'라고 하고, "
        "가장 최근 이슈의 번호와 등록일시(KST)를 알려주세요.\n"
        "5. 관련 데이터가 일부라도 있으면 그 내용을 답하고, 완전히 없을 때만 '데이터에서 확인할 수 없습니다'라고 하세요.\n"
        "6. 질문에 직접 답하고, 역질문하지 마세요."
    )

    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(history.recent_ollama_messages(n=4))

    if context:
        user_content = f"[참고 데이터]\n{context}\n\n[질문] {query}"
    else:
        user_content = f"[참고 데이터 없음]\n[질문] {query}\n데이터에서 확인할 수 없습니다."
    messages.append({"role": "user", "content": user_content})

    answer = _call_messages(messages, model=model)
    return answer, sources


def free_chat(query: str, history: ChatHistory, model: str | None = None) -> str:
    """자유 대화 모드: RAG 없이 multi-turn 대화."""
    system = (
        "당신은 친절하고 유능한 어시스턴트입니다.\n"
        "규칙:\n"
        "1. 한국어로 자연스럽게 답변하세요.\n"
        "2. 사용자의 질문에 직접 답변하세요. 질문으로 되묻지 마세요.\n"
        "3. 게임 QA, 버그 분석, 데이터 분석 분야에 특히 전문성이 있습니다.\n"
        "4. 모든 주제에 성실하게 답변하되, 확실하지 않은 내용은 솔직하게 밝히세요.\n"
        "5. 이전 대화 내용을 참고하여 맥락에 맞게 답변하세요."
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(history.recent_ollama_messages(n=6))
    messages.append({"role": "user", "content": query})
    return _call_messages(messages, model=model)
