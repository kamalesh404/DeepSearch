"""Streamlit dashboard: search UI, results view, and knowledge graph visualization."""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.middleware import API_KEYS  # noqa: E402
from core.config import get_config  # noqa: E402
from core.engine import SearchEngine  # noqa: E402
from graph.knowledge import KnowledgeGraphBuilder  # noqa: E402
from graph.query import GraphQueryEngine  # noqa: E402

st.set_page_config(page_title="DeepSearch", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading search engine…")
def load_engine() -> SearchEngine:
    """Build the engine once per process and cache it across reruns."""
    return SearchEngine()


@st.cache_data(show_spinner="Building knowledge graph…")
def build_graph(corpus: tuple[str, ...]) -> dict:
    """Construct a subgraph around entities found in the cached corpus."""
    builder = KnowledgeGraphBuilder()
    graph = builder.build(list(corpus))
    return GraphQueryEngine(graph).subgraph(
        seed_entities=list(graph.entities)[:12]
    )


def render_sidebar() -> None:
    """Show configuration summary and API key hint."""
    cfg = get_config()
    with st.sidebar:
        st.header("Settings")
        st.caption(f"top_k={cfg.get('search.top_k')} · min_score={cfg.get('search.min_score')}")
        st.divider()
        demo_key = sorted(API_KEYS)[0] if API_KEYS else "n/a"
        st.text_input("Demo API key", value=demo_key, disabled=True)
        st.caption("Pass as X-API-Key header on /search calls.")


def render_results(query: str) -> None:
    """Execute the query and lay out hits plus optional graph panel."""
    engine = load_engine()
    response = engine.search(query)

    col_results, col_graph = st.columns([3, 2])
    with col_results:
        st.subheader(f"{response.total_hits} results · {response.total_ms:.0f} ms")
        if not response.results:
            st.info("No documents matched. Try indexing content via POST /index.")
        for hit in response.results:
            with st.container(border=True):
                st.markdown(f"**{hit.title or hit.doc_id}**  ·  score `{hit.score}`")
                st.write(hit.snippet)
                if hit.url:
                    st.caption(hit.url)

    with col_graph:
        st.subheader("Knowledge graph")
        corpus = tuple(hit.title + " " + hit.snippet for hit in response.results)
        try:
            import networkx as nx  # optional visualization dependency

            data = build_graph(corpus) if corpus else {"nodes": [], "edges": []}
            g = nx.DiGraph()
            g.add_nodes_from((n["id"], n) for n in data.get("nodes", []))
            g.add_edges_from((e["source"], e["target"], e) for e in data.get("edges", []))
            if len(g):
                st.graphviz_chart(nx.nx_agraph.to_agraph(g))
            else:
                st.caption("Graph is empty.")
        except ImportError:
            st.caption("Install networkx+pygraphviz to visualize the graph.")


def main() -> None:
    """Compose the dashboard page."""
    st.title("DeepSearch 🔍")
    render_sidebar()
    query = st.text_input("Search", placeholder="Ask anything… e.g. \"bm25\" ranking")
    go = st.button("Search", type="primary")
    if query and go:
        render_results(query)


if __name__ == "__main__":
    main()
