"""
app.py
──────
Streamlit UI for the Multi-Agent Research & Report Generation System.

Features:
- Query input with examples
- Private document upload + ingestion
- Real-time agent progress tracking
- Live report display with syntax highlighting
- Report download
- Quality evaluation display
"""

import streamlit as st
import tempfile
import os
import time
from pathlib import Path

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports after page config ─────────────────────────────────────────────────
from core.streaming import stream_graph, AGENT_LABELS, AGENT_DESCRIPTIONS
from core.evaluation import evaluate_run
from tools.rag_tool import ingest_documents

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e2e8f0;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* Agent status cards */
    .agent-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .agent-card.running {
        border-color: #3b82f6;
        background: #1e3a5f;
    }
    .agent-card.done {
        border-color: #22c55e;
        background: #14532d22;
    }
    .agent-card.error {
        border-color: #ef4444;
        background: #7f1d1d22;
    }

    /* Score cards */
    .score-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #334155;
    }
    .score-value {
        font-size: 2rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .score-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* Report area */
    .report-container {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 2rem;
        max-height: 70vh;
        overflow-y: auto;
    }

    /* Sidebar */
    .sidebar-section {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
AGENT_ORDER = [
    "researcher",
    "fact_checker",
    "summarizer",
    "analyst",
    "critic",
    "report_writer",
]

STATUS_ICONS = {
    "pending": "⏳",
    "running": "🔄",
    "done":    "✅",
    "error":   "❌",
}

EXAMPLE_QUERIES = [
    "What are the economic and geopolitical implications of the global AI chip shortage?",
    "How is quantum computing expected to impact cryptography and cybersecurity by 2030?",
    "Analyze the causes, effects, and solutions to the global water scarcity crisis.",
    "What are the latest breakthroughs in CRISPR gene editing and their ethical implications?",
    "How is remote work transforming urban real estate markets globally?",
]


# ── Session state initialisation ──────────────────────────────────────────────
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {a: "pending" for a in AGENT_ORDER}
if "final_report" not in st.session_state:
    st.session_state.final_report = ""
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "critique_count" not in st.session_state:
    st.session_state.critique_count = 0
if "quality_score" not in st.session_state:
    st.session_state.quality_score = 0.0
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 📄 Private Documents (RAG)")
    uploaded_files = st.file_uploader(
        "Upload PDFs, DOCX, or TXT files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Uploaded documents will be ingested into the private RAG store and searched alongside web results."
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", type="secondary", use_container_width=True):
            with st.spinner("Ingesting documents into vector store..."):
                saved_paths = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    for f in uploaded_files:
                        path = os.path.join(tmpdir, f.name)
                        with open(path, "wb") as fp:
                            fp.write(f.getvalue())
                        saved_paths.append(path)

                    count = ingest_documents(saved_paths)
                    st.session_state.ingested_files.extend([f.name for f in uploaded_files])

            st.success(f"✅ Ingested {count} chunks from {len(uploaded_files)} file(s)")

    if st.session_state.ingested_files:
        st.markdown("**Ingested files:**")
        for fname in st.session_state.ingested_files[-5:]:
            st.markdown(f"  📎 {fname}")

    st.divider()

    st.markdown("### 📋 Agent Pipeline")
    agent_status_placeholder = st.empty()

    def render_agent_status():
        status_text = ""
        for agent in AGENT_ORDER:
            icon = STATUS_ICONS.get(st.session_state.agent_status.get(agent, "pending"), "⏳")
            label = AGENT_LABELS.get(agent, agent)
            status = st.session_state.agent_status.get(agent, "pending")
            status_text += f"{icon} {label} — `{status}`\n\n"
        agent_status_placeholder.markdown(status_text)

    render_agent_status()

    st.divider()

    if st.session_state.final_state:
        st.markdown("### 📊 Quality Metrics")
        eval_result = evaluate_run(st.session_state.final_state)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall", f"{eval_result.overall_score}/10")
            st.metric("Coverage", f"{eval_result.coverage_score}/10")
        with col2:
            st.metric("Grade", eval_result.grade)
            st.metric("Sources", eval_result.source_count)
        st.metric("Critique Loops", eval_result.critique_loops)
        st.caption(eval_result.summary)


# ── Main area ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 Multi-Agent Research System</h1>
    <p>Powered by LangGraph · OpenAI · ChromaDB · Tavily</p>
</div>
""", unsafe_allow_html=True)

# Query input section
col_query, col_btn = st.columns([4, 1])

with col_query:
    query = st.text_area(
        "Research Query",
        placeholder="Enter your research question here... (e.g., 'What are the economic implications of AI automation on the global workforce?')",
        height=100,
        label_visibility="collapsed",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacer
    run_button = st.button(
        "🚀 Research",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_running,
    )

# Example queries
st.markdown("**💡 Try an example:**")
example_cols = st.columns(len(EXAMPLE_QUERIES[:3]))
for i, (col, example) in enumerate(zip(example_cols, EXAMPLE_QUERIES[:3])):
    with col:
        if st.button(
            example[:50] + "...",
            key=f"example_{i}",
            use_container_width=True,
            help=example,
        ):
            st.session_state.example_query = example
            st.rerun()

# Handle example query selection
if "example_query" in st.session_state:
    query = st.session_state.pop("example_query")

st.divider()

# ── Research execution ────────────────────────────────────────────────────────

if run_button and query.strip():
    # Reset state
    st.session_state.is_running = True
    st.session_state.final_report = ""
    st.session_state.final_state = None
    st.session_state.agent_status = {a: "pending" for a in AGENT_ORDER}
    st.session_state.critique_count = 0

    # Layout for progress + report
    progress_col, report_col = st.columns([1, 2])

    with progress_col:
        st.markdown("### 🔄 Agent Progress")
        progress_bar = st.progress(0)
        current_agent_display = st.empty()
        loop_display = st.empty()
        log_expander = st.expander("📜 Live Log", expanded=False)
        log_container = log_expander.empty()
        log_entries = []

    with report_col:
        st.markdown("### 📄 Live Report Stream")
        report_display = st.empty()
        report_display.info("⏳ Waiting for report generation to begin...")

    try:
        completed_agents = 0

        for event in stream_graph(query.strip()):
            node = event.get("node", "")
            state_update = event.get("state", {}) or {}
            event_type = event.get("type", "")

            if event_type == "error":
                st.error(f"❌ Error: {event.get('error', 'Unknown error')}")
                break

            # Update agent status
            if node in AGENT_ORDER:
                agent_statuses = state_update.get("agent_status", {})
                if agent_statuses:
                    st.session_state.agent_status.update(agent_statuses)

                # Mark current as running if not yet done
                current_status = st.session_state.agent_status.get(node, "pending")
                if current_status == "pending":
                    st.session_state.agent_status[node] = "running"

                # Update sidebar agent status
                render_agent_status()

                # Update progress
                if state_update.get("agent_status", {}).get(node) == "done":
                    completed_agents = sum(
                        1 for s in st.session_state.agent_status.values()
                        if s == "done"
                    )
                    progress_bar.progress(completed_agents / len(AGENT_ORDER))

                # Current agent display
                label = AGENT_LABELS.get(node, node)
                desc = AGENT_DESCRIPTIONS.get(node, "")
                current_agent_display.markdown(
                    f"**Current:** {label}\n\n_{desc}_"
                )

                # Critique loop tracking
                if node == "critic":
                    loops = state_update.get("critique_loop_count", 0)
                    approved = state_update.get("approved", False)
                    st.session_state.critique_count = loops
                    loop_display.markdown(
                        f"**Critique Loop:** {loops} | "
                        f"**Status:** {'✅ Approved' if approved else '🔄 Reviewing'}"
                    )

                # Add to log
                log_entries.append(f"[{node}] {event.get('description', 'Processing...')}")
                log_container.markdown("\n\n".join(f"• {e}" for e in log_entries[-10:]))

            # Live report streaming
            partial_report = state_update.get("final_report_markdown", "")
            if partial_report:
                st.session_state.final_report = partial_report
                report_display.markdown(partial_report[:3000] + ("..." if len(partial_report) > 3000 else ""))

            # Capture final state
            if state_update.get("final_report_markdown"):
                st.session_state.final_state = state_update

        # Mark all remaining as done
        for agent in AGENT_ORDER:
            if st.session_state.agent_status.get(agent) != "done":
                st.session_state.agent_status[agent] = "done"

        progress_bar.progress(1.0)
        current_agent_display.success("✅ Research complete!")
        render_agent_status()

    except Exception as e:
        st.error(f"❌ Research failed: {str(e)}")
        st.exception(e)
    finally:
        st.session_state.is_running = False

elif not query.strip() and run_button:
    st.warning("⚠️ Please enter a research query.")


# ── Final Report Display ──────────────────────────────────────────────────────
if st.session_state.final_report:
    st.divider()
    st.markdown("## 📄 Final Research Report")

    # Quality scores
    if st.session_state.final_state:
        try:
            eval_result = evaluate_run(st.session_state.final_state)
            score_cols = st.columns(5)
            metrics = [
                ("Overall", eval_result.overall_score, "/10"),
                ("Grade", eval_result.grade, ""),
                ("Sources", eval_result.source_count, " used"),
                ("Loops", eval_result.critique_loops, " critique(s)"),
                ("Words", eval_result.word_count, " words"),
            ]
            for col, (label, value, suffix) in zip(score_cols, metrics):
                col.metric(label, f"{value}{suffix}")
        except Exception:
            pass

    # Report tabs
    tab_rendered, tab_raw = st.tabs(["📖 Rendered Report", "📝 Raw Markdown"])

    with tab_rendered:
        st.markdown(st.session_state.final_report)

    with tab_raw:
        st.code(st.session_state.final_report, language="markdown")

    # Download button
    st.download_button(
        label="⬇️ Download Report (.md)",
        data=st.session_state.final_report,
        file_name=f"research_report_{int(time.time())}.md",
        mime="text/markdown",
        type="primary",
    )