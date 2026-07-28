"""
dashboard.py - QA 통합 대시보드

Streamlit 기반 게임 QA 웹 대시보드.
- GitLab 이슈를 불러와 AI(Ollama)로 QA 체크리스트 자동 생성
- CSV 테이블 키워드 검색 및 AI 보조 데이터 검증
- 테이블 뷰어, AI 챗봇(RAG / 자유대화) 등 QA 업무 기능 통합 제공
"""

import os, re, time, json
from datetime import datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from search import (
    ai_search,
    multi_keyword_search,
    load_all_tables,
    load_table_metadata,
    parse_table_descriptions,
    get_available_model,
    ai_validate_data,
    ai_select_tables,
    extract_specific_tokens,
    programmatic_validate,
    ai_generate_checklist,
    fetch_gitlab_issue,
    CSV_DIR,
)
from chatbot import (
    RAGIndex,
    ChatHistory,
    rag_chat,
    free_chat,
    fetch_gitlab_issues_bulk,
    get_chat_model,
)

HOME_PATH      = os.path.join(os.path.dirname(__file__), "home.md")
WIKI_PATH      = os.path.join(os.path.dirname(__file__), "wiki.md")
COMMANDS_PATH  = os.path.join(os.path.dirname(__file__), "commands.md")
TABLE_PATH     = os.path.join(os.path.dirname(__file__), "table.md")
CHEAT_EXE_PATH   = os.path.join(os.path.dirname(__file__), "dist", "QA_CheatTool.exe")
CHEAT_NOTES_PATH = os.path.join(os.path.dirname(__file__), "cheat_notes.md")

# ──────────────────────────────────────────────
st.set_page_config(page_title="Game QA Tool", page_icon="🎮",
                   layout="wide", initial_sidebar_state="expanded")

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Michroma&family=Inter:wght@800&display=swap');

.stApp { background:#0a0c14; }
section[data-testid="stSidebar"] { background:#0d1017 !important; border-right:1px solid #181f2c; }

.qa-header {
    font-family:'Michroma',sans-serif; font-size:1.75rem; color:#d8eaf8;
    margin:0; line-height:1; padding:10px 0 14px; letter-spacing:0.04em;
}
.page-header {
    font-family:'Michroma',sans-serif; font-size:1.2rem; color:#a0bcd8;
    margin:0; line-height:1; padding:8px 0 12px; letter-spacing:0.06em;
    border-bottom:1px solid #182535; margin-bottom:16px;
}
.sec-label {
    font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#4a6888;
    text-transform:uppercase; letter-spacing:0.14em; margin:0 0 6px;
}
.stat-card {
    background:#0f1420; border:1px solid #1a2535; border-radius:8px;
    padding:12px 14px; text-align:center;
}
.stat-number { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:#6a88a8; line-height:1; }
.stat-label  { font-size:0.62rem; color:#3d5570; text-transform:uppercase; letter-spacing:0.1em; margin-top:3px; }

.tbl-header {
    display:flex; align-items:baseline; gap:10px;
    padding:8px 14px; background:#0c1420;
    border-left:2px solid #304a64; border-radius:6px 6px 0 0; margin-top:18px;
}
.tbl-name  { font-family:'JetBrains Mono',monospace; font-weight:700; color:#7aa0c0; font-size:0.82rem; }
.tbl-count { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#3a5570; background:#0d1828; padding:1px 8px; border-radius:10px; }
.tbl-desc  { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#52789a; margin-left:auto; }

.kw-tag  { display:inline-block; background:#0c1828; border:1px solid #1a3050; color:#486080;
           font-family:'JetBrains Mono',monospace; font-size:0.7rem; padding:2px 9px; border-radius:20px; margin:2px; }
.tgt-tag { display:inline-block; background:#0c1f12; border:1px solid #1a3d20; color:#3d7848;
           font-family:'JetBrains Mono',monospace; font-size:0.7rem; padding:2px 9px; border-radius:20px; margin:2px; }

.validation-box {
    background:#0c1420; border:1px solid #253545; border-radius:8px;
    padding:18px 22px; font-family:'JetBrains Mono',monospace;
    font-size:0.8rem; color:#80a0bc; line-height:1.8; white-space:pre-wrap;
}
.info-box {
    background:#09131e; border-left:2px solid #253d52; border-radius:5px;
    padding:12px 16px; font-family:'JetBrains Mono',monospace;
    font-size:0.75rem; color:#607898; line-height:1.7; margin-bottom:16px;
}
.error-box {
    background:#140a0a; border:1px solid #351515; border-radius:7px;
    padding:12px 16px; color:#8a4848; font-family:'JetBrains Mono',monospace; font-size:0.8rem;
}
.empty-box {
    background:#0c1017; border:1px dashed #1a2535; border-radius:7px;
    padding:48px; text-align:center; color:#4a6880; font-family:'JetBrains Mono',monospace;
}

.model-ok  { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#3d6848;
             background:#09180d; padding:6px 12px; border-radius:5px; border:1px solid #183020; }
.model-err { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#7a3535;
             background:#140808; padding:6px 12px; border-radius:5px; border:1px solid #2c1010; }

.stTextInput>div>div>input,
.stTextArea>div>div>textarea {
    background:#0c1420 !important; border:1px solid #182535 !important;
    border-radius:6px !important; color:#b8cce0 !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.82rem !important;
}
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
    border-color:#304a6a !important; box-shadow:0 0 0 2px rgba(48,74,106,0.25) !important;
}
.stButton>button {
    background:#0e1c2e !important; color:#607898 !important;
    border:1px solid #1c3248 !important; border-radius:6px !important;
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.76rem !important; font-weight:600 !important;
    padding:6px 14px !important; transition:all 0.13s !important;
}
.stButton>button:hover { background:#132238 !important; color:#90b4d4 !important; border-color:#285070 !important; }
.stButton>button[kind="primary"] { background:#122030 !important; color:#80aed0 !important; border-color:#224060 !important; }
.stTabs [data-baseweb="tab"] {
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.76rem !important; color:#252f3d !important; letter-spacing:0.06em !important;
}
.stTabs [aria-selected="true"] { color:#6a90b0 !important; }
.stCheckbox label { font-family:'JetBrains Mono',monospace !important; font-size:0.76rem !important; color:#304055 !important; }
.stSelectbox>div>div, .stMultiSelect>div>div {
    background:#0c1420 !important; border:1px solid #182535 !important;
    border-radius:6px !important; color:#b8cce0 !important;
    font-family:'JetBrains Mono',monospace !important;
}
.stDataFrame { border:1px solid #182535 !important; border-radius:0 0 6px 6px !important; }
hr { border-color:#121c28 !important; }

::-webkit-scrollbar { height: 10px; width: 10px; }
::-webkit-scrollbar-track { background: #0c1828; border-radius: 5px; }
::-webkit-scrollbar-thumb { background: #2a4a6a; border-radius: 5px; min-width: 40px; min-height: 40px; }
::-webkit-scrollbar-thumb:hover { background: #3d6090; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────
_defaults = {
    "page":              "홈",
    "qa_docs_open":      False,
    "tables_cache":      None,
    "table_descs":       {},
    "search_results":    {},
    "last_query":        "",
    "expanded_keywords": [],
    "target_tables":     [],
    "search_time":       0.0,
    "rag_index":         None,
    "chat_history":      None,
    "chat_model":        None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
def md_table_to_tsv(md_text: str) -> str:
    """Markdown 테이블을 Google Sheets 붙여넣기용 TSV로 변환."""
    rows = []
    for line in md_text.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r'^\|[\s\-:|]+\|$', line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append("\t".join(cells))
    return "\n".join(rows)


def render_md_page(title: str, icon: str, md_path: str, edit_key: str, default: str,
                   title_size: str = "1.2rem"):
    col_title, col_edit = st.columns([5, 1])
    with col_title:
        st.markdown(f'<div class="page-header" style="font-size:{title_size};">{icon} {title}</div>',
                    unsafe_allow_html=True)
    with col_edit:
        edit_mode = st.toggle("편집 모드", value=False, key=f"{edit_key}_toggle")
    content = Path(md_path).read_text(encoding="utf-8") if Path(md_path).exists() else default
    if edit_mode:
        edited = st.text_area("편집", value=content, height=650,
                              label_visibility="collapsed", key=f"{edit_key}_editor")
        if st.button("💾  저장", type="primary", key=f"{edit_key}_save"):
            Path(md_path).write_text(edited, encoding="utf-8")
            st.success("저장되었습니다.")
            st.rerun()
    else:
        st.markdown(content)


def render_df(df: pd.DataFrame, height: int = 420, head: int = None):
    data = df.head(head) if head else df
    st.dataframe(data, use_container_width=True, hide_index=True, height=height)


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 0 18px;">
        <div style="font-family:'Michroma',sans-serif; color:#90b8d8; font-size:0.95rem; letter-spacing:0.06em;">
            🎮 QA SEARCH
        </div>
        <div style="font-family:'JetBrains Mono',monospace; color:#2e3d50; font-size:0.58rem;
             margin-top:3px; letter-spacing:0.14em; text-transform:uppercase;">Game Data Tool</div>
    </div>""", unsafe_allow_html=True)

    # 페이지 네비게이션
    st.markdown('<div class="sec-label">페이지</div>', unsafe_allow_html=True)

    _qa_docs_sub = {"명령어 목록", "테이블 설명", "팀 구조"}
    _is_in_docs  = st.session_state.page in _qa_docs_sub

    for label, key in [("🏠  홈", "홈"), ("🤖  AI 챗봇", "AI 챗봇"), ("🗂  QA 도구", "QA 도구")]:
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.rerun()

    # QA문서 상위 메뉴
    _docs_arrow = "▼" if st.session_state.qa_docs_open else "▶"
    if st.button(f"{_docs_arrow}  📚  QA 문서", key="nav_qa_docs", use_container_width=True,
                 type="primary" if (_is_in_docs or st.session_state.qa_docs_open) else "secondary"):
        st.session_state.qa_docs_open = not st.session_state.qa_docs_open
        st.rerun()

    if st.session_state.qa_docs_open:
        _sub_options = ["💻  명령어 목록", "📋  테이블 설명", "👥  팀 구조"]
        _sub_keys    = ["명령어 목록",     "테이블 설명",     "팀 구조"]
        _cur_idx = _sub_keys.index(st.session_state.page) if st.session_state.page in _sub_keys else None

        def _on_sub_radio():
            idx = _sub_options.index(st.session_state.qa_docs_radio)
            st.session_state.page = _sub_keys[idx]

        st.radio("", _sub_options, index=_cur_idx,
                 label_visibility="collapsed",
                 key="qa_docs_radio", on_change=_on_sub_radio)

    if st.button("📥  다운로드", key="nav_다운로드", use_container_width=True,
                 type="primary" if st.session_state.page == "다운로드" else "secondary"):
        st.session_state.page = "다운로드"
        st.rerun()

    st.markdown("---")

    if st.button("🔄  테이블 새로고침", use_container_width=True):
        st.session_state.tables_cache = None
        st.session_state.table_descs  = {}
        st.rerun()

    if st.session_state.tables_cache is None:
        if Path(CSV_DIR).exists():
            with st.spinner("CSV 로딩 중..."):
                st.session_state.tables_cache = load_all_tables(CSV_DIR)
        else:
            st.session_state.tables_cache = {}

    if not st.session_state.table_descs:
        st.session_state.table_descs = parse_table_descriptions(load_table_metadata())

    tables      = st.session_state.tables_cache
    table_descs = st.session_state.table_descs

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(tables)}</div>'
                    f'<div class="stat-label">테이블</div></div>', unsafe_allow_html=True)
    with c2:
        total_rows = sum(len(d) for d in tables.values())
        disp = f"{total_rows//1000}k" if total_rows >= 1000 else str(total_rows)
        st.markdown(f'<div class="stat-card"><div class="stat-number">{disp}</div>'
                    f'<div class="stat-label">총 행수</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="sec-label">검색 설정</div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#5a7898;
         line-height:1.7; margin-bottom:10px;">
        <b style="color:#7aa0c0;">AI 테이블 선별</b> — 키워드 분석 후<br>
        &nbsp;&nbsp;관련 테이블만 골라 검색 범위 축소<br>
        <b style="color:#7aa0c0;">최대 결과 행수</b> — 테이블당 최대 출력 행
    </div>""", unsafe_allow_html=True)
    use_ai       = st.checkbox("AI 테이블 선별 + 키워드 확장", value=True)
    max_rows_opt = st.selectbox("최대 결과 행수", [50, 100, 200, 500], index=1,
                                label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="sec-label">AI 모델</div>', unsafe_allow_html=True)
    try:
        _model = get_available_model()
        st.markdown(f'<div class="model-ok">✓ &nbsp;{_model}</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="model-err">✗ &nbsp;Ollama 연결 실패</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
st.markdown('<div class="qa-header">GAME QA SEARCH</div>', unsafe_allow_html=True)

current_page = st.session_state.page


# ══════════════════════════════════════════════
# PAGE 1 : QA 도구
# ══════════════════════════════════════════════
if current_page == "QA 도구":
    tab_checklist, tab_search, tab_validate, tab_viewer = st.tabs([
        "📝  체크리스트 생성",
        "🔍  테이블 검색",
        "✅  데이터 검증",
        "📊  테이블 뷰어",
    ])

    # ── 테이블 검색 ──
    with tab_search:
        with st.form(key="search_form", clear_on_submit=False):
            col_inp, col_btn = st.columns([5, 1])
            with col_inp:
                query = st.text_input("검색어",
                    placeholder="아이템 CID, 퀘스트명, 이벤트ID 등... (예: 10001, 불꽃검, 포션)",
                    label_visibility="collapsed")
            with col_btn:
                search_clicked = st.form_submit_button("검색", use_container_width=True)

        if search_clicked and query.strip():
            if not tables:
                st.markdown('<div class="error-box">❌ CSV 파일 없음. 사이드바에서 새로고침하세요.</div>',
                            unsafe_allow_html=True)
            else:
                t0 = time.time()
                with st.spinner("검색 중..."):
                    if use_ai:
                        result = ai_search(query, csv_dir=CSV_DIR, use_ai_expand=True,
                                           preloaded_tables=tables)
                        st.session_state.expanded_keywords = result.get("keywords", [query])
                        st.session_state.target_tables     = result.get("target_tables", [])
                        st.session_state.search_results    = result.get("results", {})
                        if result.get("error"):
                            st.markdown(f'<div class="error-box">❌ {result["error"]}</div>',
                                        unsafe_allow_html=True)
                    else:
                        _q       = query.strip()
                        _tokens  = extract_specific_tokens(_q)
                        _kws     = list(dict.fromkeys(
                            _tokens + [_q] + ([_q.replace(" ", "")] if " " in _q else [])
                        ))
                        st.session_state.search_results    = multi_keyword_search(tables, _kws, max_rows_opt)
                        st.session_state.expanded_keywords = _kws
                        st.session_state.target_tables     = []
                st.session_state.last_query  = query
                st.session_state.search_time = time.time() - t0

        results = st.session_state.search_results
        if results:
            total_found = sum(len(df) for df in results.values())
            kw_tags  = " ".join(f'<span class="kw-tag">{k}</span>' for k in st.session_state.expanded_keywords)
            tgt_tags = " ".join(f'<span class="tgt-tag">{t}</span>' for t in st.session_state.target_tables)
            st.markdown(f"""
            <div style="margin:14px 0 5px; font-size:0.7rem; color:#4a6880;">
                결과 <span style="color:#6a90b0;">{total_found}행</span>
                &nbsp;/&nbsp;<span style="color:#486480;">{len(results)}테이블</span>
                &nbsp;·&nbsp;<span style="color:#3a5570;">{st.session_state.search_time:.2f}s</span>
            </div>
            <div style="margin-bottom:3px; font-size:0.65rem; color:#3a5570;">키워드</div>
            <div style="margin-bottom:6px;">{kw_tags}</div>""", unsafe_allow_html=True)
            if tgt_tags:
                st.markdown(f'<div style="margin-bottom:3px; font-size:0.65rem; color:#3a5570;">선별된 테이블</div>'
                            f'<div style="margin-bottom:6px;">{tgt_tags}</div>', unsafe_allow_html=True)
            for tname, df in results.items():
                desc   = table_descs.get(tname, "")
                n_cols = len(df.columns)
                st.markdown(f"""<div class="tbl-header">
                    <span class="tbl-name">📄 {tname}</span>
                    <span class="tbl-count">{len(df)} rows · {n_cols} cols</span>
                    {f'<span class="tbl-desc">{desc}</span>' if desc else ""}
                </div>""", unsafe_allow_html=True)
                render_df(df, height=min(420, (len(df)+1)*35+40))
                csv_dl = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(f"⬇  {tname} 다운로드", csv_dl,
                                   f"{tname}_결과.csv", "text/csv", key=f"dl_{tname}")
        elif st.session_state.last_query:
            st.markdown("""<div class="empty-box">
                <div style="font-size:1.6rem; margin-bottom:8px; opacity:0.35;">🔍</div>
                <div>검색 결과 없음</div>
                <div style="font-size:0.7rem; margin-top:5px; color:#1a2535;">다른 키워드나 CID 번호로 시도해보세요</div>
            </div>""", unsafe_allow_html=True)

    # ── 데이터 검증 ──
    with tab_validate:
        st.markdown("""<div class="info-box">
            💡 기획 변경 내용(텍스트 또는 파일)을 입력하면 AI가 관련 테이블을 자동으로 찾아 검증합니다.<br>
            &nbsp;&nbsp;&nbsp;특정 테이블을 직접 지정하고 싶으면 아래에서 선택하세요 (선택 시 자동 탐색 대신 사용).
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-bottom:6px;">검증 대상 테이블 (선택 안 하면 AI가 자동 탐색)</div>',
                    unsafe_allow_html=True)
        val_selected_tables = st.multiselect(
            "테이블 선택", sorted(tables.keys()) if tables else [],
            placeholder="비워두면 AI가 자동으로 테이블을 찾습니다",
            label_visibility="collapsed", key="val_tables")
        if val_selected_tables:
            for tsel in val_selected_tables:
                d = table_descs.get(tsel, "")
                if d:
                    st.markdown(f'<div style="font-family:JetBrains Mono,monospace; font-size:0.7rem; '
                                f'color:#4a6880; margin:2px 0;">· {tsel}: {d}</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        col_cond, col_file = st.columns([3, 2])
        with col_cond:
            st.markdown('<div class="sec-label" style="margin-bottom:6px;">검증 조건 (텍스트)</div>',
                        unsafe_allow_html=True)
            val_condition_text = st.text_area("검증 조건",
                placeholder="예: 아이템 10001번의 ATK 값이 500으로 변경되어야 합니다.",
                label_visibility="collapsed", height=120, key="val_condition_text")
        with col_file:
            st.markdown('<div class="sec-label" style="margin-bottom:6px;">검증 조건 (파일 첨부)</div>',
                        unsafe_allow_html=True)
            uploaded_cond = st.file_uploader("파일 업로드", type=["csv","pdf","txt","xlsx","md"],
                                             label_visibility="collapsed", key="val_condition_file")
            file_condition_text = ""
            condition_df        = None
            if uploaded_cond:
                ftype = uploaded_cond.name.split(".")[-1].lower()
                try:
                    if ftype in ("txt", "md"):
                        file_condition_text = uploaded_cond.read().decode("utf-8", errors="ignore")
                    elif ftype == "csv":
                        condition_df        = pd.read_csv(uploaded_cond, encoding="utf-8-sig")
                        file_condition_text = condition_df.to_string(index=False)
                    elif ftype == "xlsx":
                        condition_df        = pd.read_excel(uploaded_cond)
                        file_condition_text = condition_df.to_string(index=False)
                    elif ftype == "pdf":
                        try:
                            import pdfplumber
                            with pdfplumber.open(uploaded_cond) as pdf:
                                file_condition_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                        except ImportError:
                            st.warning("PDF 파싱: pip install pdfplumber")
                except Exception as e:
                    st.warning(f"파일 읽기 실패: {e}")
                if file_condition_text:
                    if condition_df is not None:
                        st.markdown(
                            f'<div style="font-family:JetBrains Mono,monospace; font-size:0.68rem; color:#3d6848; '
                            f'background:#09180d; padding:6px 12px; border-radius:5px; margin-top:6px; border:1px solid #183020;">'
                            f'📊 {len(condition_df)}행 · {len(condition_df.columns)}열 로드됨 — '
                            f'키 컬럼: <b>{condition_df.columns[0]}</b> / '
                            f'검사 컬럼: {", ".join(str(c) for c in condition_df.columns[1:])}</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div style="font-family:JetBrains Mono,monospace; font-size:0.68rem; color:#2e4458; '
                            f'background:#09131e; padding:8px 12px; border-radius:5px; margin-top:6px; '
                            f'max-height:80px; overflow:auto; white-space:pre-wrap;">'
                            f'{file_condition_text[:300]}{"..." if len(file_condition_text)>300 else ""}</div>',
                            unsafe_allow_html=True)

        val_condition_combined = "\n".join(filter(None, [val_condition_text.strip(), file_condition_text.strip()]))

        if st.button("🤖  검증 실행", type="primary", key="val_btn"):
            if not val_condition_combined:
                st.warning("검증 조건을 텍스트 또는 파일로 입력해주세요.")
            elif not tables:
                st.error("CSV 테이블을 먼저 로드해주세요.")
            else:
                if val_selected_tables:
                    final_tables = val_selected_tables
                    auto_selected = False
                else:
                    with st.spinner("AI가 검증 대상 테이블을 분석 중..."):
                        _meta = load_table_metadata()
                        final_tables = ai_select_tables(
                            val_condition_combined, _meta, list(tables.keys())
                        )
                    auto_selected = True

                if not final_tables:
                    st.warning("관련 테이블을 찾지 못했습니다. 테이블을 직접 선택해주세요.")
                    st.stop()

                if auto_selected:
                    _tags = " ".join(
                        f'<span class="tgt-tag">{t}</span>' for t in final_tables
                    )
                    st.markdown(
                        f'<div style="margin-bottom:10px; font-size:0.68rem; color:#3a5570;">'
                        f'🤖 AI 자동 선택 테이블 &nbsp;{_tags}</div>',
                        unsafe_allow_html=True)

                val_tables_subset = {t: tables[t] for t in final_tables if t in tables}

                # CSV/XLSX: 컬럼 값 직접 비교
                if condition_df is not None:
                    with st.spinner("데이터 비교 중..."):
                        prog_results = programmatic_validate(condition_df, val_tables_subset)

                    if not prog_results:
                        st.warning(f"조건 파일의 키 컬럼 `{condition_df.columns[0]}`이 대상 테이블에 존재하지 않습니다.")
                    else:
                        total_rows     = sum(r["total"]      for r in prog_results)
                        total_mismatch = sum(r["mismatches"] for r in prog_results)

                        if total_mismatch == 0:
                            st.success(f"✅ 전체 일치 — {total_rows}행 전부 일치합니다.")
                        else:
                            st.error(f"❌ 불일치 발견 — {total_rows}행 중 {total_mismatch}행이 다릅니다.")

                        for r in prog_results:
                            match_cnt = r["total"] - r["mismatches"]
                            st.markdown(
                                f'<div class="tbl-header">'
                                f'<span class="tbl-name">📄 {r["table"]}</span>'
                                f'<span class="tbl-count">키: {r["key_col"]} · 검사: {", ".join(r["checked_cols"])}</span>'
                                f'<span class="tbl-count">참 {match_cnt} / 거짓 {r["mismatches"]}</span>'
                                f'</div>', unsafe_allow_html=True)
                            if r["summary"]:
                                for line in r["summary"]:
                                    st.markdown(f"- {line}")
                            else:
                                st.markdown("모든 행이 일치합니다.")

                        if total_mismatch > 0:
                            with st.spinner("AI가 불일치 원인을 분석하고 있습니다..."):
                                ai_comment = ai_validate_data(
                                    condition=val_condition_combined,
                                    search_results=val_tables_subset,
                                )
                            st.markdown("#### AI 불일치 분석")
                            st.markdown(f'<div class="validation-box">{ai_comment}</div>', unsafe_allow_html=True)

                # 텍스트 조건: AI 검증
                else:
                    with st.spinner("AI가 데이터를 검증하고 있습니다..."):
                        val_result_text = ai_validate_data(
                            condition=val_condition_combined,
                            search_results=val_tables_subset,
                        )
                    st.markdown("#### AI 검증 결과")
                    st.markdown(f'<div class="validation-box">{val_result_text}</div>', unsafe_allow_html=True)
                    st.markdown("#### 참조 데이터")
                    for tname, df in val_tables_subset.items():
                        desc   = table_descs.get(tname, "")
                        n_cols = len(df.columns)
                        st.markdown(f"""<div class="tbl-header">
                            <span class="tbl-name">📄 {tname}</span>
                            <span class="tbl-count">{len(df)} rows · {n_cols} cols</span>
                            {f'<span class="tbl-desc">{desc}</span>' if desc else ""}
                        </div>""", unsafe_allow_html=True)
                        render_df(df, head=100)

    # ── 테이블 뷰어 ──
    with tab_viewer:
        if not tables:
            st.markdown("""<div class="empty-box">
                <div style="font-size:1.6rem; margin-bottom:8px; opacity:0.35;">📂</div>
                <div>로드된 테이블이 없습니다</div>
                <div style="font-size:0.7rem; margin-top:5px;">사이드바에서 새로고침하세요</div>
            </div>""", unsafe_allow_html=True)
        else:
            col_sel, col_search, col_info = st.columns([3, 3, 2])
            with col_search:
                viewer_filter = st.text_input("테이블명 검색", placeholder="테이블명으로 필터...",
                                              label_visibility="collapsed", key="viewer_filter")
            viewer_names = sorted(tables.keys())
            if viewer_filter:
                viewer_names = [n for n in viewer_names if viewer_filter.lower() in n.lower()]
            with col_sel:
                selected_table = st.selectbox("테이블 선택",
                    viewer_names if viewer_names else ["(검색 결과 없음)"],
                    label_visibility="collapsed", key="viewer_table_sel")

            if selected_table and selected_table in tables:
                df   = tables[selected_table]
                desc = table_descs.get(selected_table, "")
                with col_info:
                    st.markdown(f"""<div style="font-size:0.7rem; color:#252f3d; margin-top:8px;
                        font-family:'JetBrains Mono',monospace;">
                        행 <span style="color:#5a80a0;">{len(df):,}</span>
                        &nbsp;·&nbsp; 열 <span style="color:#486070;">{len(df.columns)}</span>
                    </div>""", unsafe_allow_html=True)
                if desc:
                    st.markdown(
                        f'<div style="font-family:JetBrains Mono,monospace; font-size:0.72rem; '
                        f'color:#2e4458; margin:6px 0 10px; border-left:2px solid #1e3448; '
                        f'padding-left:10px;">{desc}</div>', unsafe_allow_html=True)

                viewer_kw = st.text_input("데이터 검색 (키워드)", placeholder="테이블 내 키워드 검색...",
                                          label_visibility="collapsed", key="viewer_kw")
                if viewer_kw.strip():
                    _vkw = viewer_kw.strip()
                    mask = df.apply(lambda c: c.astype(str).str.contains(
                        _vkw, case=False, na=False, regex=False)).any(axis=1)
                    view_df = df[mask].reset_index(drop=True)
                    st.markdown(f'<div style="font-family:JetBrains Mono,monospace; font-size:0.68rem; '
                                f'color:#3a5878; margin:4px 0 6px;">검색 결과: {len(view_df)}행</div>',
                                unsafe_allow_html=True)
                else:
                    view_df = df

                with st.expander("컬럼 정보"):
                    col_info_df = pd.DataFrame({
                        "컬럼명": df.columns,
                        "타입":   [str(df[c].dtype) for c in df.columns],
                        "샘플값": [str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else ""
                                  for c in df.columns],
                    })
                    st.dataframe(col_info_df, use_container_width=True, hide_index=True)

                render_df(view_df, height=520)
                csv_dl = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(f"⬇  {selected_table}.csv 다운로드",
                                   csv_dl, f"{selected_table}.csv", "text/csv", key="viewer_dl")


    # ── 체크리스트 생성 ──
    with tab_checklist:
        st.markdown("""<div class="info-box">
            💡 GitLab 이슈 번호를 입력하면 본문·코멘트를 자동으로 수집해 체크리스트를 생성합니다.<br>
            &nbsp;&nbsp;&nbsp;이슈 없이 직접 내용을 입력하거나 파일을 첨부해도 됩니다. 여러 입력을 합산해서 분석합니다.
        </div>""", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown('<div class="sec-label" style="margin-bottom:6px;">GitLab 이슈 번호 (선택)</div>',
                        unsafe_allow_html=True)
            gl_issue_number = st.text_input(
                "이슈 번호",
                placeholder="예: 1234",
                label_visibility="collapsed",
                key="gl_issue_number",
            )
            st.markdown('<div class="sec-label" style="margin-bottom:6px; margin-top:10px;">파일 첨부 (선택)</div>',
                        unsafe_allow_html=True)
            uploaded_spec = st.file_uploader(
                "파일 업로드",
                type=["pdf", "txt", "md"],
                label_visibility="collapsed",
                key="checklist_file",
            )
            file_spec_text = ""
            if uploaded_spec:
                ftype = uploaded_spec.name.split(".")[-1].lower()
                try:
                    if ftype in ("txt", "md"):
                        file_spec_text = uploaded_spec.read().decode("utf-8", errors="ignore")
                    elif ftype == "pdf":
                        try:
                            import pdfplumber
                            with pdfplumber.open(uploaded_spec) as pdf:
                                file_spec_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                        except ImportError:
                            st.warning("PDF 파싱: pip install pdfplumber")
                except Exception as e:
                    st.warning(f"파일 읽기 실패: {e}")
                if file_spec_text:
                    st.markdown(
                        f'<div style="font-family:JetBrains Mono,monospace; font-size:0.68rem; color:#3d6848; '
                        f'background:#09180d; padding:6px 12px; border-radius:5px; margin-top:6px; border:1px solid #183020;">'
                        f'📄 {uploaded_spec.name} — {len(file_spec_text)}자 로드됨</div>',
                        unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="sec-label" style="margin-bottom:6px;">추가 내용 입력 (선택)</div>',
                        unsafe_allow_html=True)
            extra_input = st.text_area(
                "추가 내용",
                placeholder="기획서 내용이나 보충 설명을 직접 입력하세요...",
                label_visibility="collapsed",
                height=100,
                key="checklist_extra",
            )

        if st.button("🤖  체크리스트 생성", type="primary", key="checklist_btn"):
            if not gl_issue_number.strip() and not extra_input.strip() and not file_spec_text.strip():
                st.warning("이슈 번호, 내용 입력, 파일 첨부 중 하나 이상 필요합니다.")
            else:
                spec_parts = []

                if gl_issue_number.strip():
                    with st.spinner("GitLab 이슈 및 코멘트 불러오는 중..."):
                        gl_data = fetch_gitlab_issue(gl_issue_number.strip())
                    if "error" in gl_data:
                        st.error(gl_data["error"])
                        st.stop()

                    spec_parts.append(f"[이슈 제목]\n{gl_data['title']}")
                    if gl_data.get("labels"):
                        spec_parts.append(f"[레이블]\n{', '.join(gl_data['labels'])}")
                    if gl_data.get("description"):
                        spec_parts.append(f"[이슈 본문]\n{gl_data['description']}")
                    for i, c in enumerate(gl_data.get("comments", []), 1):
                        spec_parts.append(f"[코멘트 {i} - {c['author']}]\n{c['body']}")

                if file_spec_text.strip():
                    spec_parts.append(f"[첨부 파일 내용]\n{file_spec_text.strip()}")

                if extra_input.strip():
                    spec_parts.append(f"[추가 내용]\n{extra_input.strip()}")

                full_spec = "\n\n".join(spec_parts)
                with st.spinner("AI가 체크리스트를 작성하고 있습니다..."):
                    checklist_result = ai_generate_checklist(full_spec)
                st.session_state["checklist_result"] = checklist_result

        if st.session_state.get("checklist_result"):
            result_md = st.session_state["checklist_result"]
            st.markdown("#### 생성된 체크리스트")
            st.markdown(result_md)
            tsv = md_table_to_tsv(result_md)
            tsv_json = json.dumps(tsv)
            components.html(f"""
<script id="tsv-data" type="application/json">{tsv_json}</script>
<button id="copy-btn" onclick="
  const data = JSON.parse(document.getElementById('tsv-data').textContent);
  navigator.clipboard.writeText(data).then(() => {{
    this.textContent = '✓ 복사됨!';
    this.style.color = '#3d7848';
    this.style.borderColor = '#1a3d20';
    setTimeout(() => {{
      this.textContent = '📋  Google Sheets 복사';
      this.style.color = '';
      this.style.borderColor = '';
    }}, 2000);
  }});
" style="background:#0e1c2e; color:#607898; border:1px solid #1c3248;
         border-radius:6px; font-family:monospace; font-size:12px;
         font-weight:600; padding:6px 14px; cursor:pointer;">
  📋  Google Sheets 복사
</button>
""", height=45)


# ══════════════════════════════════════════════
# PAGE 2 : AI 챗봇
# ══════════════════════════════════════════════
elif current_page == "AI 챗봇":
    st.markdown('<div class="page-header">🤖 AI 챗봇</div>', unsafe_allow_html=True)

    if st.session_state.chat_history is None:
        st.session_state.chat_history = ChatHistory()
    if st.session_state.rag_index is None:
        st.session_state.rag_index = RAGIndex()
    if "chat_use_rag" not in st.session_state:
        st.session_state.chat_use_rag = False

    history: ChatHistory = st.session_state.chat_history
    index:   RAGIndex    = st.session_state.rag_index
    if st.session_state.chat_model is None:
        st.session_state.chat_model = get_chat_model()
    _chat_model = st.session_state.chat_model

    _MODE_FREE = "💬  자유대화"
    _MODE_DATA = "📂  데이터 참조 모드"

    def _on_mode_change():
        st.session_state.chat_use_rag = (st.session_state.chat_mode_radio == _MODE_DATA)

    st.radio(
        "", [_MODE_FREE, _MODE_DATA],
        index=1 if st.session_state.chat_use_rag else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="chat_mode_radio",
        on_change=_on_mode_change,
    )
    _use_rag = st.session_state.chat_use_rag

    if _use_rag:
        col_idx, col_gl = st.columns([1, 1])
        with col_idx:
            if st.button("📊  테이블 로드", use_container_width=True):
                with st.spinner("테이블 로드 중..."):
                    index.clear()
                    index.add_tables(tables)
                    index.build()
                st.success(f"✅ 테이블 {index.size}개 청크 로드 완료")
        with col_gl:
            if st.button("🦊  GitLab 로드", use_container_width=True):
                with st.spinner("GitLab 이슈 조회 중..."):
                    issues = fetch_gitlab_issues_bulk(max_issues=100)
                    if issues:
                        index.clear_gitlab()
                        index.add_gitlab_issues(issues)
                        index.build()
                        st.success(f"✅ GitLab {len(issues)}개 이슈 로드 완료")
                    else:
                        st.warning("GitLab 이슈를 불러올 수 없습니다. .env 설정을 확인하세요.")
        if index.size == 0:
            st.info("💡 테이블 로드 또는 GitLab 로드 후 질문하세요.")

    for msg in history.messages:
        role    = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        with st.chat_message(role):
            st.markdown(content)
            if sources:
                src_tags = "  ".join(f"`{s}`" for s in sources)
                st.caption(f"📎 출처: {src_tags}")

    if history.messages:
        if st.button("🗑  대화 초기화", key="chat_clear_btn"):
            history.clear()
            st.rerun()

    user_input = st.chat_input("질문을 입력하세요 (예: Item 10001의 BuyPrice는?)")

    if user_input and user_input.strip():
        q = user_input.strip()

        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("생성 중..."):
                if _use_rag:
                    answer, sources = rag_chat(q, index, history, model=_chat_model)
                else:
                    answer  = free_chat(q, history, model=_chat_model)
                    sources = []
            st.markdown(answer)
            if sources:
                st.caption("📎 출처: " + "  ".join(f"`{s}`" for s in sources))

        # LLM 응답 완료 후 히스토리 추가 (사전 추가 시 현재 질문이 컨텍스트에 중복됨)
        history.add("user", q)
        history.add("assistant", answer, sources)
        st.rerun()


# ══════════════════════════════════════════════
# PAGE 3 : 홈
# ══════════════════════════════════════════════
elif current_page == "홈":
    render_md_page("홈", "🏠", HOME_PATH, "home", "# 🔗 유용한 링크\n\n링크를 추가해주세요.")


# ══════════════════════════════════════════════
# PAGE 4 : 팀 구조
# ══════════════════════════════════════════════
elif current_page == "팀 구조":
    render_md_page("팀 구조", "👥", WIKI_PATH, "wiki", "# 팀 구조\n\nwiki.md 파일을 생성해주세요.")


# ══════════════════════════════════════════════
# PAGE 5 : 명령어 목록
# ══════════════════════════════════════════════
elif current_page == "명령어 목록":
    render_md_page("명령어 목록", "💻", COMMANDS_PATH, "commands", "# 📌 명령어 목록\n\ncommands.md 파일을 생성해주세요.")


# ══════════════════════════════════════════════
# PAGE 6 : 테이블 설명
# ══════════════════════════════════════════════
elif current_page == "테이블 설명":
    render_md_page("테이블 설명", "📋", TABLE_PATH, "table", "# 테이블 설명\n\ntable.md 파일을 생성해주세요.")


# ══════════════════════════════════════════════
# PAGE 7 : 다운로드
# ══════════════════════════════════════════════
elif current_page == "다운로드":
    st.markdown('<div class="page-header">📥 다운로드</div>', unsafe_allow_html=True)

    (tab_cheat,) = st.tabs(["⚙  치트툴"])

    with tab_cheat:
        _exe = Path(CHEAT_EXE_PATH)
        if _exe.exists():
            _size_mb = _exe.stat().st_size / (1024 * 1024)
            _mtime   = datetime.fromtimestamp(_exe.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.markdown(f"""
            <div style="background:#0d1117; border:1px solid #1a2535; border-radius:8px;
                        padding:18px 22px; margin-bottom:12px;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.9rem;
                            color:#7aa0c0; font-weight:700; margin-bottom:6px;">
                    ⚙ &nbsp;QA CheatTool
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#3a5570; line-height:1.8;">
                    치트 명령어 실행 &nbsp;·&nbsp; 매크로 녹화 / 재생 &nbsp;·&nbsp; 반복 실행<br>
                    <span style="color:#2a4a62;">{_size_mb:.1f} MB &nbsp;|&nbsp; 최종 빌드 &nbsp;{_mtime}</span>
                </div>
            </div>""", unsafe_allow_html=True)
            with open(_exe, "rb") as _f:
                st.download_button(
                    label="⬇  QA_CheatTool.exe 다운로드",
                    data=_f,
                    file_name="QA_CheatTool.exe",
                    mime="application/octet-stream",
                )
        else:
            st.markdown("""
            <div style="background:#0d1117; border:1px dashed #1a2535; border-radius:8px;
                        padding:16px 22px; font-family:'JetBrains Mono',monospace;
                        font-size:0.75rem; color:#3a5570;">
                ⚙ &nbsp;EXE 파일 없음 &nbsp;— &nbsp;build_cheat.bat 실행 후 dist/ 폴더를 확인하세요.
            </div>""", unsafe_allow_html=True)

        render_md_page("업데이트 내역", "📋", CHEAT_NOTES_PATH, "cheat_notes",
                       "# 📋 업데이트 내역\n\n업데이트 내용을 입력해주세요.\n\n## v1.0\n- 최초 배포",
                       title_size="1.5rem")


# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'JetBrains Mono',monospace;
     font-size:0.58rem; color:#253545; padding:5px 0;">
    GAME QA SEARCH TOOL · OLLAMA + STREAMLIT
</div>""", unsafe_allow_html=True)
