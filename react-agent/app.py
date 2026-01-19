import os
import textwrap

import streamlit as st
from dotenv import load_dotenv

# LangChain / tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# =========================
# Environment & LLM setup
# =========================

load_dotenv()  # optional, loads GOOGLE_API_KEY and SERPAPI_API_KEY from .env if present

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY", "YOUR_SERPAPI_API_KEY_HERE")

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",  # light, fast model
    temperature=0.0,              # deterministic for consistent results
)

serp = SerpAPIWrapper()  # for Search tool


# =========================
# Tool 1: Search
# =========================

def search_tool(query: str, num_results: int = 5) -> str:
    """
    Use SerpAPI to search the web and return a compact, LLM-friendly summary
    of the top results.
    """
    try:
        raw_results = serp.results(query, num_results=num_results)
    except TypeError:
        raw_results = serp.results(query)

    organic = raw_results.get("organic_results", [])
    if not organic:
        return serp.run(query)

    lines = [f"Search results for: {query}"]
    for i, item in enumerate(organic[:num_results], start=1):
        title = item.get("title") or "No title"
        snippet = item.get("snippet") or item.get("snippet_highlighted_words") or ""
        link = item.get("link") or item.get("url") or ""
        lines.append(f"{i}. {title}\n   Snippet: {snippet}\n   URL: {link}")

    return "\n".join(lines)


search_tool_for_agent = Tool(
    name="Search",
    func=search_tool,
    description=(
        "Use this tool to search the web for up-to-date information. "
        "Input should be a natural language search query. "
        "The tool returns a numbered list of results with titles, snippets, and URLs."
    ),
)


# =========================
# Tool 2: Compare
# =========================

def _parse_compare_query(query: str):
    """
    Parse a comparison query of the form:
        'item1, item2, ..., category'
    Returns (items_list, category) or (None, error_message).
    """
    parts = [p.strip() for p in query.split(",") if p.strip()]

    if len(parts) < 3:
        return None, (
            "Invalid input for Compare tool. "
            "Please provide at least two items and one category, e.g.: "
            "'iPhone 15, Pixel 9, battery life'."
        )

    *items, category = parts

    if len(items) < 2:
        return None, (
            "Invalid input for Compare tool. "
            "Please provide at least two items before the category."
        )

    return (items, category), None


COMPARE_PROMPT_TEMPLATE = """
You are an expert comparison assistant.

Your task:
- Compare the following items on the category: "{category}".

Items:
{items_block}

Please:
1. Briefly describe each item (if possible).
2. Compare them specifically on "{category}" (strengths & weaknesses).
3. End with a short, clear recommendation: which item is best for "{category}" and why.

Format:
- Use short headings or bullet points.
- Be concise but informative.
"""


def compare_items(query: str) -> str:
    """
    Compare multiple items on a given category.

    Expected input format:
        "item1, item2, ..., category"
    """
    parsed, error = _parse_compare_query(query)
    if error:
        return error

    items, category = parsed
    items_block = "\n".join(f"- {item}" for item in items)

    prompt = COMPARE_PROMPT_TEMPLATE.format(
        category=category,
        items_block=items_block,
    )

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        return f"Compare tool failed to generate a comparison. Error: {e}"

    if hasattr(response, "content"):
        return response.content
    else:
        return str(response)


compare_tool_for_agent = Tool(
    name="Compare",
    func=compare_items,
    description=(
        "Use this to compare multiple items along a single category. "
        "Input format: 'item1, item2, ..., category'. "
        "Example: 'iPhone 15, Pixel 9, battery life'. "
        "The tool returns a concise, structured comparison and recommendation."
    ),
)


# =========================
# Tool 3: Analyze
# =========================

ANALYZE_PROMPT_TEMPLATE = """
You are an expert analysis assistant.

Your job is to help a user with the following question:
"{query}"

You are given the following information (which may come from web search results or a comparison of items):

---
{results}
---

Please:
1. Summarize the most important points relevant to the user's question.
2. Highlight any trade-offs, pros/cons, or key insights.
3. If appropriate, make a clear recommendation or conclusion.
4. Keep the answer concise and easy to read (short paragraphs or bullet points).

Do NOT just repeat the text; extract and organize the key information for the user.
"""


def analyze_results(results: str, query: str) -> str:
    prompt = ANALYZE_PROMPT_TEMPLATE.format(
        query=query,
        results=results,
    )
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        return f"Analyze tool failed to generate an analysis. Error: {e}"

    if hasattr(response, "content"):
        return response.content
    else:
        return str(response)


def analyze_tool(input_text: str) -> str:
    """
    Wrapper to make the analysis function easy for the ReAct agent to use.

    Formats:
    - "query ||| results"
    - or just "results" (query will be 'Unknown question')
    """
    if "|||" in input_text:
        query, results = input_text.split("|||", 1)
        query = query.strip()
        results = results.strip()
    else:
        query = "Unknown question"
        results = input_text.strip()

    return analyze_results(results=results, query=query)


analyze_tool_for_agent = Tool(
    name="Analyze",
    func=analyze_tool,
    description=(
        "Use this to summarize and extract key information from search results or "
        "comparison outputs. "
        "Input can be either the raw text to analyze, or 'user question ||| text to analyze' "
        "for more context."
    ),
)


# =========================
# Agent builder
# =========================

TOOLS = [
    search_tool_for_agent,
    compare_tool_for_agent,
    analyze_tool_for_agent,
]


def build_agent(max_iterations: int):
    """
    Create a ReAct agent with the given max_iterations.
    We keep verbose=False and show reasoning ourselves in the UI.
    """
    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,                # we'll show reasoning via intermediate_steps
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        early_stopping_method="generate",
        return_intermediate_steps=True,
    )
    return agent


def format_observation(obs, max_len: int = 800) -> str:
    """
    Shorten long observations for display.
    """
    text = str(obs)
    return textwrap.shorten(text, width=max_len, placeholder=" ... [truncated]")


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="ReAct Agent Demo", page_icon="🤖", layout="centered")

st.title("📱 ReAct Agent with Search, Compare, and Analyze")

st.markdown(
    "Ask a complex question that might require web search, comparison, and analysis. "
    "The agent will decide which tools to use and in what order."
)

# Sidebar controls
st.sidebar.header("Agent Settings")

max_iterations = st.sidebar.selectbox(
    "Max reasoning steps (iterations)",
    options=[3, 4, 5, 6],
    index=1,  # default = 4
    help="Maximum number of Thought → Action → Observation cycles.",
)

show_reasoning = st.sidebar.checkbox(
    "Show step-by-step reasoning",
    value=True,
    help="Display the agent's Thoughts, Actions, and Observations.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**LLM:** gemini-flash-latest")
st.sidebar.markdown("**Tools:** Search, Compare, Analyze")

# Main input area
user_query = st.text_area(
    "Enter your query:",
    value="What are the top 3 smartphones in 2023, and how do they compare in terms of camera quality and battery life?",
    height=120,
)

run_button = st.button("Run ReAct Agent")

# When the user clicks the button
if run_button:
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Thinking with tools..."):
            agent = build_agent(max_iterations=max_iterations)
            try:
                result = agent.invoke({"input": user_query})
            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
                st.stop()

        # Final answer
        final_answer = result.get("output", result)
        st.subheader("✅ Final Answer")
        st.write(final_answer)

        # Optional reasoning trace
        if show_reasoning:
            st.subheader("🧠 Step-by-Step Reasoning (ReAct Trace)")

            intermediate_steps = result.get("intermediate_steps", [])
            if not intermediate_steps:
                st.info("No intermediate steps available.")
            else:
                for i, (action, observation) in enumerate(intermediate_steps, start=1):
                    with st.expander(f"Step {i}: {action.tool}", expanded=False):
                        # action.log often contains the Thought + Action lines
                        st.markdown("**Raw Agent Log:**")
                        st.code(action.log)

                        st.markdown("**Tool Input:**")
                        st.write(action.tool_input)

                        st.markdown("**Observation:**")
                        st.write(format_observation(observation))
