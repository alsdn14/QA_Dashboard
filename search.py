"""
search.py - Ollama 기반 AI 검색 엔진

로컬 AI 모델(Ollama)을 활용한 게임 데이터 검색·검증 모듈.
- CSV 테이블 자동 로드 및 인코딩 감지
- 다중 키워드 OR 검색 (공백 제거 매칭 포함)
- AI 기반 관련 테이블 자동 선별 및 키워드 확장
- 기획 조건과 실제 데이터 비교 검증
- GitLab 이슈 연동 및 QA 체크리스트 자동 생성
"""

import os
import json
import re
import time
import functools
import pandas as pd
import chardet
import ollama
from pathlib import Path

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
# CSV 파일 디렉토리 — 실행 환경에 맞게 경로를 설정하세요
CSV_DIR = os.path.join(os.path.dirname(__file__), "data", "csv")
TABLE_MD = os.path.join(os.path.dirname(__file__), "table.md")

# 모델 우선순위 (설치된 모델 중 가장 먼저 발견된 모델 사용)
PREFERRED_MODELS = [
    "gemma2:9b",       # 속도/품질 균형 우수
    "gemma3:4b",       # 경량, 빠름
    "llama3.2:3b",     # 경량
    "qwen2.5:7b",      # 대안
]


# ──────────────────────────────────────────────
# CSV 로딩
# ──────────────────────────────────────────────
def detect_encoding(filepath: str) -> str:
    """파일 인코딩 자동 감지 (한글 깨짐 방지)."""
    with open(filepath, "rb") as f:
        raw = f.read(50000)
    detected = chardet.detect(raw)
    enc = detected.get("encoding", "utf-8") or "utf-8"
    if enc.lower() in ("euc-kr", "cp949", "johab"):
        return "cp949"
    if enc.lower() in ("utf-8-sig", "utf-8"):
        return "utf-8-sig"
    return enc


def load_csv(filepath: str) -> pd.DataFrame | None:
    """CSV 파일을 한글 깨짐 없이 로드."""
    encodings_to_try = [detect_encoding(filepath), "utf-8-sig", "cp949", "utf-8", "euc-kr"]
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            return df
        except Exception:
            continue
    return None


def load_all_tables(csv_dir: str) -> dict[str, pd.DataFrame]:
    """디렉토리 내 모든 CSV를 로드해 {파일명: DataFrame} 딕셔너리로 반환."""
    tables = {}
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        return tables
    for f in csv_path.glob("*.csv"):
        df = load_csv(str(f))
        if df is not None and not df.empty:
            tables[f.stem] = df
    return tables


@functools.lru_cache(maxsize=1)
def load_table_metadata() -> str:
    """table.md에서 테이블 설명 로드 (프로세스 수명 동안 캐싱)."""
    if os.path.exists(TABLE_MD):
        with open(TABLE_MD, "r", encoding="utf-8") as f:
            return f.read()
    return "테이블 메타데이터 없음"


# ──────────────────────────────────────────────
# Ollama 유틸
# ──────────────────────────────────────────────
_model_cache: dict = {"model": None, "ts": 0.0}

def get_available_model() -> str:
    """설치된 Ollama 모델 중 우선순위에 따라 모델 반환 (60초 캐싱)."""
    if _model_cache["model"] and time.time() - _model_cache["ts"] < 60:
        return _model_cache["model"]
    try:
        installed = [m.model for m in ollama.list().models]
        result = installed[0] if installed else "gemma2:9b"
        for preferred in PREFERRED_MODELS:
            for inst in installed:
                if inst.startswith(preferred.split(":")[0]):
                    result = inst
                    break
            else:
                continue
            break
    except Exception:
        result = "gemma2:9b"
    _model_cache["model"] = result
    _model_cache["ts"] = time.time()
    return result


def call_ollama(prompt: str, system: str = "", model: str = None) -> str:
    """Ollama API 호출."""
    if model is None:
        model = get_available_model()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = ollama.chat(model=model, messages=messages)
        return resp.message.content.strip()
    except Exception as e:
        return f"[Ollama 오류] {e}"


# ──────────────────────────────────────────────
# 키워드 검색 (AI 없는 빠른 검색)
# ──────────────────────────────────────────────

def multi_keyword_search(tables: dict, keywords: list[str], max_rows: int = 100) -> dict[str, pd.DataFrame]:
    """여러 키워드를 OR 조건으로 전체 테이블 1회 스캔 검색 (누락 방지)."""
    results = {}
    kws = [str(k).strip().lower() for k in keywords if str(k).strip()]
    if not kws:
        return results
    pattern = "|".join(re.escape(k) for k in kws)
    # 공백 제거 패턴: "섬광화살" → "섬광 화살" 데이터도 인식
    nospace_kws = list(dict.fromkeys(k.replace(" ", "") for k in kws))
    nospace_pattern = "|".join(re.escape(k) for k in nospace_kws)

    for name, df in tables.items():
        try:
            mask = df.apply(
                lambda col: col.astype(str).str.lower().str.contains(pattern, na=False, regex=True)
            ).any(axis=1)
            nospace_mask = df.apply(
                lambda col: col.astype(str).str.replace(" ", "", regex=False).str.lower()
                             .str.contains(nospace_pattern, na=False, regex=True)
            ).any(axis=1)
            matched = df[mask | nospace_mask].head(max_rows)
            if not matched.empty:
                results[name] = matched.reset_index(drop=True)
        except Exception:
            continue
    return results


def extract_specific_tokens(query: str) -> list[str]:
    """자연어 쿼리에서 CID·ID 등 구체적 식별자 추출 (숫자, 따옴표 구문)."""
    tokens = []
    # 5자리 이상 숫자 → CID·ID일 가능성 높음
    long_nums = re.findall(r'\b\d{5,}\b', query)
    tokens.extend(long_nums)
    # 따옴표로 감싼 정확한 구문
    quoted = re.findall(r'["\'](.+?)["\']', query)
    tokens.extend(quoted)
    # 긴 숫자가 없을 때만 2~4자리 숫자도 포함
    if not long_nums:
        tokens.extend(re.findall(r'\b\d{2,4}\b', query))
    return list(dict.fromkeys(tokens))


def parse_table_descriptions(table_metadata: str) -> dict[str, str]:
    """
    table.md 마크다운 테이블에서 파일명 → 설명 매핑 파싱.
    반환: { "Item": "아이템 마스터 테이블", "ItemConsume": "소모성 아이템 효과", ... }
    """
    desc_map: dict[str, str] = {}
    for line in table_metadata.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        name_match = re.match(r'`([^`]+)`', cells[0])
        if not name_match:
            continue
        raw_name = name_match.group(1)
        stem = raw_name.replace(".csv", "").replace(".CSV", "").strip()
        desc = cells[1].strip()
        if not desc or set(desc.replace(" ", "").replace("-", "").replace(":", "")) == set():
            continue
        if stem and desc and "---" not in desc:
            desc_map[stem] = desc
    return desc_map


def ai_select_tables(user_query: str, table_metadata: str, available_tables: list[str]) -> list[str]:
    """
    AI가 table.md를 참고해 검색에 필요한 테이블만 선별.
    반환: 테이블 파일명 리스트 (확장자 없음)
    """
    system = (
        "당신은 게임 QA 데이터 전문가입니다. "
        "사용자 검색어와 테이블 명세를 보고, 검색에 관련된 테이블 파일명만 JSON 배열로 반환하세요. "
        "파일명은 확장자(.csv) 없이 정확히 반환하세요. 예: [\"Item\", \"ItemConsume\"]"
    )
    prompt = f"""
테이블 명세:
{table_metadata}

현재 로드된 테이블 목록:
{json.dumps(available_tables, ensure_ascii=False)}

사용자 검색어: "{user_query}"

위 검색어와 직접 관련된 테이블만 골라서 JSON 배열로 반환하세요.
관련 없는 테이블은 절대 포함하지 마세요. 최대 5개.
"""
    raw = call_ollama(prompt, system)
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            selected = json.loads(match.group())
            valid = [t for t in selected if t in available_tables]
            if valid:
                return valid
        except Exception:
            pass
    # fallback: 쿼리와 이름이 겹치는 테이블 우선
    kw_lower = user_query.lower()
    fallback = [
        t for t in available_tables
        if kw_lower in t.lower() or t.lower() in kw_lower
    ]
    return fallback[:5] if fallback else available_tables


# ──────────────────────────────────────────────
# AI 검증: 데이터 정적 테스트
# ──────────────────────────────────────────────
def programmatic_validate(condition_df: pd.DataFrame, actual_tables: dict) -> list:
    """
    조건 DataFrame의 첫 번째 컬럼을 키로 삼아 실제 테이블과 행/컬럼 비교.
    - 조건 파일에서 모든 값이 비어있는 컬럼은 검사 제외
    - 셀 단위로 빈 값이면 해당 셀 비교 스킵
    """
    results = []
    key_col  = str(condition_df.columns[0])
    raw_chk  = [str(c) for c in condition_df.columns[1:]]

    non_empty_chk = [
        c for c in raw_chk
        if not condition_df[c].isna().all()
        and not (condition_df[c].astype(str).str.strip() == "").all()
    ]

    for tname, actual_df in actual_tables.items():
        actual_col_strs = [str(c) for c in actual_df.columns]
        if key_col not in actual_col_strs:
            continue
        valid_chk = [c for c in non_empty_chk if c in actual_col_strs]
        if not valid_chk:
            continue

        rows = []
        for _, cond_row in condition_df.iterrows():
            key_val = cond_row[key_col]
            matched = actual_df[actual_df[key_col].astype(str) == str(key_val)]
            entry   = {key_col: key_val}

            if matched.empty:
                entry["상태"] = "거짓 (행 없음)"
                for col in valid_chk:
                    entry[f"{col}_조건"] = cond_row[col]
                    entry[f"{col}_실제"] = "-"
            else:
                act   = matched.iloc[0]
                diffs = []
                for col in valid_chk:
                    cv_raw = cond_row[col]
                    if pd.isna(cv_raw) or str(cv_raw).strip() == "":
                        continue
                    cv = str(cv_raw).strip()
                    av = str(act[col]).strip()
                    entry[f"{col}_조건"] = cv
                    entry[f"{col}_실제"] = av
                    if cv != av:
                        diffs.append(col)
                entry["상태"] = "참" if not diffs else "거짓"
            rows.append(entry)

        diff_df  = pd.DataFrame(rows)
        mismatch = int(diff_df["상태"].str.startswith("거짓").sum())

        summary_lines = []
        for _, row in diff_df.iterrows():
            status  = str(row.get("상태", ""))
            key_val = row[key_col]
            if status == "참":
                continue
            if "행 없음" in status:
                summary_lines.append(f"**{key_col}={key_val}** 행은 실제 테이블에 존재하지 않으므로 틀렸습니다.")
            else:
                diff_parts = []
                for col in valid_chk:
                    cv = str(row.get(f"{col}_조건", "")).strip()
                    av = str(row.get(f"{col}_실제", "")).strip()
                    if cv and av and cv != av and cv != "-":
                        diff_parts.append(f"{col}이(가) '{cv}'이어야 하지만 실제로는 '{av}'")
                if diff_parts:
                    summary_lines.append(
                        f"**{key_col}={key_val}** 행은 {', '.join(diff_parts)}이므로 틀렸습니다."
                    )
                else:
                    summary_lines.append(f"**{key_col}={key_val}** 행은 틀렸습니다.")

        results.append({
            "table":        tname,
            "key_col":      key_col,
            "checked_cols": valid_chk,
            "total":        len(diff_df),
            "mismatches":   mismatch,
            "diff_df":      diff_df,
            "summary":      summary_lines,
        })
    return results


def _extract_id_tokens(condition: str) -> list[str]:
    # 우선순위: 5자리↑ 숫자 → Cid/ID 키워드 뒤 숫자 → 따옴표 구문 → 첫 번째 숫자
    # \b 대신 (?<!\d)/(?!\d): 한글은 \w 취급되어 \b 오동작 방지
    long_nums = re.findall(r'(?<!\d)\d{5,}(?!\d)', condition)
    if long_nums:
        return list(dict.fromkeys(long_nums))

    # 독립된 Cid/ID 키워드 뒤 숫자 (복합어 접미 Cid는 제외)
    id_after_kw = re.findall(r'(?<!\w)(?:Cid|CID|cid|ID)[^\d]{0,5}(\d{2,})', condition)
    if id_after_kw:
        return list(dict.fromkeys(id_after_kw))

    quoted = re.findall(r'["\'](.+?)["\']', condition)
    if quoted:
        return list(dict.fromkeys(quoted))

    first = re.findall(r'(?<!\d)\d{2,}(?!\d)', condition)
    return [first[0]] if first else []


def _filter_rows_for_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    # 1순위: Cid 컬럼 정확 일치 → 2순위: 전체 컬럼 부분 문자열 매칭
    # 3순위: 한국어 고유명사 OR 매칭 → fallback: head(30)
    id_tokens = _extract_id_tokens(condition)

    if id_tokens and 'Cid' in df.columns:
        for token in id_tokens:
            try:
                exact = df[df['Cid'] == int(token)]
            except ValueError:
                exact = df[df['Cid'].astype(str) == token]
            if not exact.empty:
                return exact.head(50)

    if id_tokens:
        pattern = "|".join(re.escape(t) for t in id_tokens)
        mask = df.apply(
            lambda col: col.astype(str).str.contains(pattern, na=False, regex=True)
        ).any(axis=1)
        relevant = df[mask]
        if not relevant.empty:
            return relevant.head(50)

    kor_terms = list(dict.fromkeys(re.findall(r'[가-힣]{2,}', condition)))
    if kor_terms:
        pattern = "|".join(re.escape(t) for t in kor_terms)
        mask = df.apply(
            lambda col: col.astype(str).str.contains(pattern, na=False, regex=True)
        ).any(axis=1)
        relevant = df[mask]
        if not relevant.empty:
            return relevant.head(50)

    return df.head(30)


def ai_validate_data(
    condition: str,
    search_results: dict[str, pd.DataFrame]
) -> str:
    """기획 조건과 실제 데이터를 AI로 비교 검증. 텍스트 조건 전용."""
    table_metadata = load_table_metadata()

    id_tokens = _extract_id_tokens(condition)
    id_hint = f"검증 대상 식별자: {', '.join(id_tokens)}" if id_tokens else ""

    data_text = ""
    for tname, df in search_results.items():
        rows_to_show = _filter_rows_for_condition(df, condition)
        data_text += f"\n[{tname}]\n"
        for _, row in rows_to_show.iterrows():
            data_text += f"  Cid={row.get('Cid', '?')}:\n"
            for col, val in row.items():
                data_text += f"    {col}: {val}\n"
            data_text += "\n"

    system = (
        "당신은 게임 데이터 QA 검증 시스템입니다.\n"
        "검증 조건은 틀릴 수 있습니다. 데이터를 먼저 읽고, 마지막에 판정하세요.\n\n"
        "[출력 순서 — 반드시 이 순서로 작성]\n"
        "1. 실제값: [컬럼명]='[데이터에서 읽은 정확한 값]' 형태로 인용\n"
        "2. 비교: 실제값과 조건 기대값의 차이를 설명\n"
        "3. 판정: ✅ 일치  또는  ❌ 불일치  또는  ⚠️ 확인불가\n\n"
        "[주의]\n"
        "- BuyPrice와 SellPrice, NextQuestCids와 QuestStepCids 등은 서로 다른 컬럼입니다. 컬럼명을 정확히 읽으세요.\n"
        "- 조건의 기대값([1013], 500 등)을 데이터에서 찾은 값이라고 착각하지 마세요.\n"
        "- 실제값과 기대값이 하나라도 다르면 반드시 '판정: ❌ 불일치'로 끝내세요.\n"
        "- 데이터에서 직접 읽은 값만 사용하세요."
    )
    prompt = f"""
{id_hint}

실제 데이터:
{data_text[:6000]}

검증 조건:
{condition}

[절차]
대상 Cid/ID: {', '.join(id_tokens) if id_tokens else '조건에서 파악'}
1. Cid 컬럼이 위 ID와 정확히 일치하는 행을 선택하세요.
2. 검증할 컬럼의 실제 값을 따옴표로 인용하세요: 실제값: [컬럼명]='[값]'
3. 조건 기대값과 비교하세요.
4. 마지막 줄: 판정: ✅ 일치 / ❌ 불일치 / ⚠️ 확인불가
"""
    result = call_ollama(prompt, system)
    result = re.sub(r'\n{2,}', '\n', result.strip())
    result = re.sub(r' {2,}', ' ', result)
    return result


# ──────────────────────────────────────────────
# AI 자연어 검색 통합 함수
# ──────────────────────────────────────────────
def ai_search(
    user_query: str,
    csv_dir: str = CSV_DIR,
    use_ai_expand: bool = True,
    preloaded_tables: dict = None,
) -> dict:
    """
    메인 검색 함수.
    반환: { "keywords": [...], "target_tables": [...], "results": {테이블명: DataFrame}, "model": str, "error": str }
    """
    output = {"keywords": [], "target_tables": [], "results": {}, "model": "", "error": ""}

    tables = preloaded_tables if preloaded_tables else load_all_tables(csv_dir)
    if not tables:
        output["error"] = f"CSV 파일을 찾을 수 없습니다: {csv_dir}"
        return output

    specific_tokens = extract_specific_tokens(user_query)

    q = user_query.strip()
    q_nospace = q.replace(" ", "")
    base_keywords = list(dict.fromkeys(
        specific_tokens + ([q] if q else []) + ([q_nospace] if q_nospace != q else [])
    ))

    table_metadata = load_table_metadata()
    available_tables = list(tables.keys())

    if use_ai_expand:
        target_tables = ai_select_tables(user_query, table_metadata, available_tables)
    else:
        target_tables = []
    keywords = base_keywords

    output["keywords"]      = keywords
    output["target_tables"] = target_tables
    output["model"]         = get_available_model()

    merged = multi_keyword_search(tables, keywords)

    if specific_tokens and merged:
        filtered: dict[str, pd.DataFrame] = {}
        for tname, df in merged.items():
            mask = pd.Series([False] * len(df), index=df.index)
            for tok in specific_tokens:
                mask |= df.apply(
                    lambda col, _t=tok: col.astype(str).str.contains(re.escape(_t), na=False, regex=False)
                ).any(axis=1)
            fdf = df[mask].reset_index(drop=True)
            if not fdf.empty:
                filtered[tname] = fdf
        if filtered:
            merged = filtered

    # 정렬: 셀 완전 일치 → 부분 포함 → AI 선별 → 나머지
    exact_phrases = [p for p in [q, q_nospace] if p]
    exact_cell: dict[str, pd.DataFrame] = {}
    exact_sub:  dict[str, pd.DataFrame] = {}

    for tname, df in merged.items():
        if tname in exact_cell or tname in exact_sub:
            continue
        for phrase in exact_phrases:
            try:
                pl = phrase.lower()
                if df.apply(
                    lambda col: col.astype(str).str.strip().str.lower() == pl
                ).any(axis=1).any():
                    exact_cell[tname] = df
                    break
                if df.apply(
                    lambda col: col.astype(str).str.lower().str.contains(
                        re.escape(pl), na=False, regex=True
                    )
                ).any(axis=1).any():
                    exact_sub[tname] = df
                    break
            except Exception:
                pass

    # 운영 도구·로그성 테이블은 exact_sub 내에서 뒤로 밀기 (게임 데이터 테이블 우선)
    # 실제 프로젝트의 낮은 우선순위 테이블명으로 교체해 사용하세요
    _LOW_PRIORITY: set[str] = set()

    exact_sub_hi = {t: df for t, df in exact_sub.items() if t not in _LOW_PRIORITY}
    exact_sub_lo = {t: df for t, df in exact_sub.items() if t in _LOW_PRIORITY}
    exact_sub = {**exact_sub_hi, **exact_sub_lo}

    exact_table_names = set(exact_cell) | set(exact_sub)
    exact_first = {**exact_cell, **exact_sub}

    ai_second   = {t: merged[t] for t in (target_tables or []) if t in merged and t not in exact_table_names}
    rest        = {t: df for t, df in merged.items() if t not in exact_table_names and t not in (target_tables or [])}

    rest = dict(list(rest.items())[: 5 if exact_table_names else len(rest)])

    merged = {**exact_first, **ai_second, **rest}

    output["results"] = merged
    return output


# ──────────────────────────────────────────────
# GitLab 이슈 가져오기
# ──────────────────────────────────────────────
def _gitlab_config() -> tuple[str, str, str]:
    """(.env 또는 환경변수에서) GitLab URL·토큰·프로젝트 경로 반환."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    url     = os.environ.get("GITLAB_URL", "").rstrip("/")
    token   = os.environ.get("GITLAB_TOKEN", "")
    project = os.environ.get("GITLAB_PROJECT", "")
    return url, token, project


def fetch_gitlab_issue(issue_number: str) -> dict:
    """이슈 번호로 GitLab 이슈 본문 + 코멘트를 가져옴 (.env 자동 사용)."""
    import requests

    gitlab_url, token, project = _gitlab_config()
    if not token:
        return {"error": ".env 파일에 GITLAB_TOKEN이 설정되지 않았습니다."}
    if not project:
        return {"error": ".env 파일에 GITLAB_PROJECT가 설정되지 않았습니다."}
    if not issue_number.strip().isdigit():
        return {"error": "이슈 번호는 숫자만 입력해주세요."}

    headers  = {"PRIVATE-TOKEN": token}
    encoded  = requests.utils.quote(project, safe="")
    base_api = f"{gitlab_url}/api/v4/projects/{encoded}/issues/{issue_number.strip()}"

    try:
        resp = requests.get(base_api, headers=headers, timeout=10, verify=False)
        if resp.status_code != 200:
            return {"error": f"API 오류 {resp.status_code}: {resp.text[:300]}"}
        d = resp.json()

        notes_resp = requests.get(f"{base_api}/notes?per_page=100&sort=asc",
                                  headers=headers, timeout=10, verify=False)
        comments = []
        if notes_resp.status_code == 200:
            for note in notes_resp.json():
                if not note.get("system", False):
                    comments.append({
                        "author": note.get("author", {}).get("name", ""),
                        "body":   note.get("body", "").strip(),
                    })

        return {
            "title":       d.get("title", ""),
            "description": d.get("description", "") or "",
            "labels":      d.get("labels", []),
            "comments":    comments,
        }
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────
# AI 체크리스트 생성
# ──────────────────────────────────────────────
def ai_generate_checklist(spec_text: str) -> str:
    """기획서 또는 GitLab 이슈 텍스트를 분석해 QA 체크리스트(Markdown Table) 생성."""
    system = (
        "너는 10년 차 숙련된 게임 QA 엔지니어이자 테스트 설계 전문가야. "
        "입력된 기획서 또는 이슈 내용을 분석해 기능적 결함을 찾기 위한 상세 체크리스트를 작성해. "
        "반드시 아래 규칙을 따라:\n"
        "1. 중분류 / 소분류 / 테스트 내용 / 결과 / 비고 순서의 5개 컬럼 Markdown Table 형식으로만 출력.\n"
        "2. '결과'와 '비고' 칸은 반드시 비워둘 것.\n"
        "3. 중복되는 테스트 케이스는 제외할 것.\n"
        "4. 테스트 내용은 구체적인 행동과 예상 결과가 포함되도록 작성할 것.\n"
        "5. 데이터 테이블 구조나 수치 변경 사항이 있다면 이를 검증하는 항목을 반드시 포함할 것.\n"
        "6. 표 이외의 설명 문구는 절대 출력하지 마세요."
    )
    prompt = f"""다음 기획서 또는 깃랩 이슈 내용을 분석하여 QA 체크리스트를 작성해 주세요.

---
{spec_text}
---

중분류 / 소분류 / 테스트 내용 / 결과 / 비고 컬럼의 Markdown Table만 출력하세요."""
    return call_ollama(prompt, system)


# ──────────────────────────────────────────────
# CLI 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    query = input("검색어 입력: ").strip()
    result = ai_search(query, use_ai_expand=False)
    print(f"\n=== 검색 결과 ===")
    for tname, df in result["results"].items():
        print(f"\n[{tname}] {len(df)}행")
        print(df.head(5).to_string())
    if result["error"]:
        print(f"오류: {result['error']}")
