"""
LangGraph ReAct agent implementation.

Defines the tool-using AI agent responsible for 1) market analysis, 2) competitor 
research, 3) demand signal extraction, 4) scoring, and 5) final report generation. 
This module coordinates reasoning and calls the RAG layer when evidence is required.
"""

import os
import re
import requests
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

from rag.rag import retrieve

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    market_overview: str
    competitors: str
    demand_signals: str
    evidence_brief: str
    scoring_breakdown: str
    final_recommendation: str
    executive_summary: str
    source_links: list
    sources: str
    final_report_md: str


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def format_rag_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No internal context found."
    return "\n".join(
        f"- Source: {c['source']} (score: {c['score']:.3f})\n  {c['text']}"
        for c in chunks
    )


def llm_summarize(text: str, instruction: str, model: str = "gpt-4o-mini") -> str:
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def extract_urls(messages) -> list[str]:
    urls = []
    for msg in messages:
        if hasattr(msg, "content") and "http" in str(msg.content):
            urls += re.findall(r'https?://[^\s\)\"]+', str(msg.content))
    return urls


# ─────────────────────────────────────────────
# RAW TOOLS (used by nodes directly)
# ─────────────────────────────────────────────
def search_web_serper(query: str, k: int = 5) -> list[dict]:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": os.getenv("SERPER_API_KEY"), "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json={"q": query}, timeout=30)
    r.raise_for_status()
    return [
        {"title": i.get("title", ""), "link": i.get("link", ""), "snippet": i.get("snippet", "")}
        for i in r.json().get("organic", [])[:k]
    ]


def search_news_newsapi(query: str, k: int = 5, language: str = "de") -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "apiKey": os.getenv("NEWS_API_KEY"), "pageSize": k, "sortBy": "publishedAt", "language": language}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return [
        {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": (a.get("source") or {}).get("name", ""),
            "publishedAt": a.get("publishedAt", ""),
            "description": a.get("description", ""),
        }
        for a in r.json().get("articles", [])[:k]
    ]


def rag_lookup(query: str, k: int = 5) -> list[dict]:
    return retrieve(query, k)


# ─────────────────────────────────────────────
# LANGGRAPH TOOLS (used by ReAct agent)
# ─────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for information using Serper. Use for market data, trends, and general research."""
    results = search_web_serper(query, k=5)
    return "\n".join([f"- {r['title']}: {r['snippet']} ({r['link']})" for r in results])


@tool
def news_search(query: str) -> str:
    """Search recent news articles using NewsAPI. Use for current trends and recent developments."""
    results = search_news_newsapi(query, k=5, language="de")
    return "\n".join([f"- {r['title']}: {r['description']}" for r in results])

@tool
def targeted_search(query: str) -> str:
    """
    Perform a focused web search for a specific research question.
    Use this when you need precise data (e.g., market size, CAGR, competitors).
    """
    results = search_web_serper(query, k=5)

    if not results:
        return "No results found."

    formatted = []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        formatted.append(f"- {title}: {snippet} ({link})")

    return "\n".join(formatted)

@tool
def internal_knowledge(query: str) -> str:
    """Search internal documents stored in Pinecone. Use for scoring framework, product profile, and report structure."""
    return format_rag_context(rag_lookup(query, k=5))


# ─────────────────────────────────────────────
# REACT AGENT (Research Agent)
# ─────────────────────────────────────────────
def build_react_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return create_react_agent(llm, [web_search, news_search, targeted_search])


# ─────────────────────────────────────────────
# SYNTHESIS AGENT (Analysis Agent)
#  ─────────────────────────────────────────────
def build_synthesis_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    return create_react_agent(llm, [internal_knowledge])





# ─────────────────────────────────────────────
# NODES (Scoring, Summary, Sources, Final Report)
# ─────────────────────────────────────────────

def score_opportunity(evidence_text: str) -> str:
    scoring_chunks = retrieve(
        "scoring framework: Market Attractiveness 1-5, Competitive Intensity 1-5, Demand Signals 1-5, and Go/Explore/No-Go criteria",
        k=8
    )
    scoring_chunks = [c for c in scoring_chunks if c.get("source") == "scoring_framework.txt"]
    scoring_context = format_rag_context(scoring_chunks)

    prompt = f"""
You are assessing whether a company should enter the German market.

Use ONLY the scoring rules in INTERNAL SCORING FRAMEWORK below for definitions.
Then score each category 1–5 and provide a Go/Explore/No-Go recommendation.

INTERNAL SCORING FRAMEWORK (from scoring_framework.txt):
{scoring_context}

EVIDENCE COLLECTED (web/news/tools):
{evidence_text}

Return:
- Scores per category (1–5)
- Short rationale per category
- Final recommendation (Go / Explore / No-Go)
"""
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise market analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def scoring_node(state: AgentState) -> AgentState:
    evidence = state.get("evidence_brief", "")
    result = score_opportunity(evidence)
    state["scoring_breakdown"] = result
    state["final_recommendation"] = llm_summarize(
        result,
        instruction="Extract ONLY the final recommendation (Go / Explore / No-Go) and its one-paragraph justification from the text below. Nothing else."
    )
    return state


def executive_summary_node(state: AgentState) -> AgentState:
    combined = f"""
MARKET OVERVIEW:\n{state.get("market_overview", "")}
COMPETITORS:\n{state.get("competitors", "")}
DEMAND SIGNALS:\n{state.get("demand_signals", "")}
SCORING:\n{state.get("scoring_breakdown", "")}
"""
    state["executive_summary"] = llm_summarize(
        combined,
        instruction="You are a senior business analyst. Write a concise executive summary (max 150 words) for a market entry report on GT's Living Foods entering the German kombucha market. Highlight the key opportunity, main challenges, and final recommendation."
    )
    return state


def sources_node(state: AgentState) -> AgentState:
    unique_links = list(dict.fromkeys(state.get("source_links", [])))
    state["sources"] = "\n".join([f"- {link}" for link in unique_links])
    return state


def final_report_node(state: AgentState) -> AgentState:
    template = Path("docs/Final_report_structure.md").read_text(encoding="utf-8")
    report = (template
        .replace("{{EXEC_SUMMARY}}", state.get("executive_summary", "TBD"))
        .replace("{{MARKET_OVERVIEW}}", state.get("market_overview", "TBD"))
        .replace("{{COMPETITORS}}", state.get("competitors", "TBD"))
        .replace("{{DEMAND_SIGNALS}}", state.get("demand_signals", "TBD"))
        .replace("{{SCORING}}", state.get("scoring_breakdown", "TBD"))
        .replace("{{RECOMMENDATION}}", state.get("final_recommendation", "TBD"))
        .replace("{{SOURCES}}", state.get("sources", "TBD"))
    )
    state["final_report_md"] = report
    return state



# ─────────────────────────────────────────────
# GRAPH / NODES
# ─────────────────────────────────────────────
def build_graph():
    react_agent = build_react_agent()

    def market_analysis_node_react(state: AgentState) -> AgentState:
        result = react_agent.invoke({"messages": [("human",
            "Research the German kombucha market. Find: market size, growth rate (CAGR), key trends, and opportunities. Use web_search and news_search tools.")]})
        state["market_overview"] = result["messages"][-1].content
        state.setdefault("source_links", [])
        state["source_links"] += extract_urls(result["messages"])
        return state

    def competitor_research_node_react(state: AgentState) -> AgentState:
        result = react_agent.invoke({"messages": [("human",
            "Research the main kombucha competitors in Germany. Find: brand names, positioning, strengths, and market presence. Use web_search.")]})
        state["competitors"] = result["messages"][-1].content
        state.setdefault("source_links", [])
        state["source_links"] += extract_urls(result["messages"])
        return state

    def demand_signals_node_react(state: AgentState) -> AgentState:
        result = react_agent.invoke({"messages": [("human",
            "Find demand signals for kombucha in Germany. Look for: consumer interest, health trends, buying behavior. Use news_search and web_search.")]})
        state["demand_signals"] = result["messages"][-1].content
        state.setdefault("source_links", [])
        state["source_links"] += extract_urls(result["messages"])
        return state

    MIN_SOURCES = 15  # threshold for evidence-quality gate

    def route_after_demand_signals(state: AgentState) -> str:
        links = state.get("source_links", [])
        return "extra_research" if len(links) < MIN_SOURCES else "evidence_synthesis"

    def extra_research_node(state: AgentState) -> AgentState:
        prompt = """
We don't have enough sources yet.

Do additional targeted research on German kombucha market evidence.
Use web_search, news_search, and targeted_search.

Focus on:
- Germany market size / growth / CAGR (if possible)
- German retail availability (major retailers, online shops)
- consumer trend signals (health/fermentation, low sugar, gut health)
- recent German/EU news about kombucha / fermented drinks

Output:
1) 5–8 bullet findings (short)
2) Include links in your response whenever possible.
"""
        result = react_agent.invoke({"messages": [("human", prompt)]})

        extra_text = result["messages"][-1].content
        state["demand_signals"] = (state.get("demand_signals", "") + "\n\nADDITIONAL RESEARCH:\n" + extra_text).strip()

        state.setdefault("source_links", [])
        state["source_links"] += extract_urls(result["messages"])

        return state
    
    # Synthesis Node:
    def evidence_synthesis_node(state: AgentState) -> AgentState:
        synthesis_agent = build_synthesis_agent()

        prompt = f"""
You are an analysis agent.

Your task is to synthesize the collected research into a structured evidence brief.
Do NOT search the web. You may use internal_knowledge if necessary.

INPUT RESEARCH

MARKET OVERVIEW:
{state.get("market_overview","")}

COMPETITORS:
{state.get("competitors","")}

DEMAND SIGNALS:
{state.get("demand_signals","")}

SOURCES:
{chr(10).join(list(dict.fromkeys(state.get("source_links", []))))}

OUTPUT FORMAT (Markdown):

## Market Facts
- key size / growth insights

## Competitor Landscape
- key players and positioning

## Demand Signals
- consumer trends
- media signals
- retail presence

## Evidence Summary
- key arguments for scoring
"""

        result = synthesis_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )

        state["evidence_brief"] = result["messages"][-1].content
        return state

    graph = StateGraph(AgentState)
    graph.add_node("market_analysis", market_analysis_node_react)
    graph.add_node("competitor_research", competitor_research_node_react)
    graph.add_node("demand_signals", demand_signals_node_react)
    graph.add_node("extra_research", extra_research_node)
    graph.add_node("evidence_synthesis", evidence_synthesis_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("executive_summary", executive_summary_node)
    graph.add_node("sources", sources_node)
    graph.add_node("final_report", final_report_node)

    graph.set_entry_point("market_analysis")
    graph.add_edge("market_analysis", "competitor_research")
    graph.add_edge("competitor_research", "demand_signals")
    graph.add_conditional_edges(
        "demand_signals",
        route_after_demand_signals,
        {
            "extra_research": "extra_research",
            "evidence_synthesis": "evidence_synthesis",
        }
    )
    graph.add_edge("extra_research", "evidence_synthesis")
    graph.add_edge("evidence_synthesis", "scoring")
    graph.add_edge("scoring", "executive_summary")
    graph.add_edge("executive_summary", "sources")
    graph.add_edge("sources", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile()




# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def run(company: str = "GT's Living Foods", industry: str = "Longevity / Health Food", target_market: str = "Germany"):
    print(f"🚀 Starting autonomous market entry agent for {company}...")
    app = build_graph()
    final_state = app.invoke({})
    print("\n✅ Agent completed!")
    print("\n📄 FINAL REPORT:")
    print(final_state.get("final_report_md", "No report generated."))
    return final_state

if __name__ == "__main__":
    run()
