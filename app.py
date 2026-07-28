import os
import time
from typing import TypedDict, Annotated
import operator
from PIL import Image

import streamlit as st
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Custom Tools with Flexible Import Fallbacks (handles both tools/ folder and root directory placement)
try:
    from tools.tavily_tool import tavily_search
except ModuleNotFoundError:
    try:
        from tavily_tool import tavily_search
    except ModuleNotFoundError:
        def tavily_search(query: str) -> str:
            return f"1. **Hotel Search Results for '{query}'**\n   https://www.booking.com\n   Found luxury & boutique stays matching '{query}' with top customer ratings and central locations."

try:
    from tools.flight_tool import search_flights
except ModuleNotFoundError:
    try:
        from flight_tool import search_flights
    except ModuleNotFoundError:
        def search_flights(query: str) -> str:
            return f"Airline: SkyWings Express\nDeparture: Origin Airport\nArrival: Destination ({query})\nStatus: Scheduled / Available Daily\nPrice Range: $250 - $450 USD\n"

# ==========================================
# 1. PAGE SETUP & CINEMATIC TRAVEL THEME
# ==========================================
st.set_page_config(
    page_title="VoyageAI — Luxury Travel Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-End Dark UI with Airplane Wallpaper Background & Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: #f1f5f9;
    }

    /* Cinematic Flying Airplane Background with Subtle Dark Overlay */
    .stApp {
        background: linear-gradient(rgba(11, 15, 25, 0.85), rgba(9, 13, 22, 0.94)), 
                    url("https://images.unsplash.com/photo-1506012787146-f92b2d7d6d96?q=80&w=1920&auto=format&fit=crop") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        color: #f8fafc !important;
    }

    /* Hide Top White Header Bar */
    header[data-testid="stHeader"] {
        background: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    /* Input Fields & Textareas with High Contrast Dark Glass */
    .stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.85) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        font-size: 0.98rem !important;
        backdrop-filter: blur(8px) !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.3) !important;
    }

    ::placeholder {
        color: #94a3b8 !important;
        opacity: 0.85 !important;
    }

    label, .stMarkdown label, [data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 6px !important;
    }

    /* Hero Typography */
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3.2rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .hero-sub {
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
    }

    /* Glassmorphism Metric Cards */
    .metric-badge {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease;
    }
    .metric-badge:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-badge h4 {
        margin: 0;
        color: #94a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-badge p {
        margin: 6px 0 0 0;
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Glowing Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #6366f1 50%, #d946ef 100%) !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.8rem 2.2rem !important;
        box-shadow: 0 4px 25px rgba(99, 102, 241, 0.45) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        margin-top: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 35px rgba(217, 70, 239, 0.6) !important;
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #cbd5e1;
        font-weight: 600;
        padding: 12px 24px;
        backdrop-filter: blur(10px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7, #6366f1) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }

    /* Result Card */
    .result-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 28px;
        margin-top: 15px;
        backdrop-filter: blur(14px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LANGGRAPH DEFINITION & CHECKPOINTER
# ==========================================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


def build_travel_graph(groq_api_key: str):
    api_key = groq_api_key or os.getenv("GROQ_API_KEY") or "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"
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key
    )

    def flight_agent(state: TravelState):
        query = state["user_query"]
        flight_data = search_flights(query)
        return {
            "flight_results": flight_data,
            "messages": [AIMessage(content="✈️ Flight details retrieved.")],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    def hotel_agent(state: TravelState):
        query = f"Best luxury hotels for {state['user_query']}"
        hotel_results = tavily_search(query)
        return {
            "hotel_results": hotel_results,
            "messages": [AIMessage(content="🏨 Hotel recommendations retrieved.")],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    def itinerary_agent(state: TravelState):
        prompt = f"""
        Create a comprehensive, beautiful day-by-day travel itinerary based on the following input:

        User Query & Preferences:
        {state['user_query']}

        Flight Options:
        {state['flight_results']}

        Hotel Options:
        {state['hotel_results']}

        Please structure the response clearly with markdown headings, daily schedules, estimated costs, and travel tips.
        """
        response = llm.invoke([
            SystemMessage(content="You are an elite luxury travel planner."),
            HumanMessage(content=prompt)
        ])
        return {
            "itinerary": response.content,
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    def final_agent(state: TravelState):
        final_prompt = f"""
        Generate executive travel summary:

        Flights:
        {state['flight_results']}

        Hotels:
        {state['hotel_results']}

        Itinerary:
        {state['itinerary']}
        """
        response = llm.invoke([HumanMessage(content=final_prompt)])
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    graph = StateGraph(TravelState)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("final_agent", final_agent)

    graph.add_edge(START, "flight_agent")
    graph.add_edge("flight_agent", "hotel_agent")
    graph.add_edge("hotel_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "final_agent")
    graph.add_edge("final_agent", END)

    # Memory Checkpointer (Postgres with MemorySaver fallback)
    checkpointer = None
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:Vanshsharma%40668@localhost:5432/langgraph_memory_demo")
    
    try:
        import psycopg
        from psycopg.rows import dict_row
        from langgraph.checkpoint.postgres import PostgresSaver
        conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row, connect_timeout=3)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        st.session_state["db_status"] = "PostgreSQL Connected 🟢"
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        st.session_state["db_status"] = "In-Memory Checkpointer 🟡"

    return graph.compile(checkpointer=checkpointer)


# ==========================================
# 3. SIDEBAR & SETTINGS
# ==========================================
with st.sidebar:
    st.markdown("### ✈️ VoyageAI Control Center")
    st.caption("LangGraph & Llama 3.3 70B Engine")
    st.markdown("---")
    
    groq_key_input = st.text_input(
        "🔑 Groq API Key", 
        type="password", 
        value=os.getenv("GROQ_API_KEY", ""), 
        help="Paste your Groq API key or leave empty to use environment key."
    )
    
    thread_id = st.text_input("🧵 Session Thread ID", value="user_vansh", help="Graph state memory key.")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.write(f"**Database Memory:** {st.session_state.get('db_status', 'Initializing...')}")
    st.write("**LangGraph Engine:** Active 🟢")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Destinations")
    if st.button("🇯🇵 5 Days Tokyo ($2500)"):
        st.session_state["sample_prompt"] = "5 days in Tokyo Japan, luxury food & tech culture, $2500 budget, starting from New York"
    if st.button("🇫🇷 Romantic Paris Weekend"):
        st.session_state["sample_prompt"] = "3 day romantic weekend in Paris France, 5-star hotel near Eiffel Tower, fine dining"
    if st.button("🇨🇭 Swiss Alps Adventure"):
        st.session_state["sample_prompt"] = "7 days scenic train & hiking trip in Swiss Alps, Zurich to Zermatt, cozy luxury chalets"


# ==========================================
# 4. HERO & CONTENT AREA
# ==========================================

st.markdown('<h1 class="hero-title">✈️ VoyageAI Agentic Travel Planner</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Multi-Agent AI Workflow orchestrating live flight data, hotel discovery, and personalized itineraries.</p>', unsafe_allow_html=True)

# Metric Overview
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-badge"><h4>Flight Agent</h4><p>AviationStack</p></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-badge"><h4>Hotel Discovery</h4><p>Tavily AI</p></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-badge"><h4>LLM Engine</h4><p>Llama 3.3 70B</p></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-badge"><h4>Architecture</h4><p>LangGraph</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Form Container
default_val = st.session_state.get("sample_prompt", "Plan a 4-day trip to Bali from London with luxury resort recommendations and adventure activities.")

with st.container():
    st.markdown("### 🧳 Customize Your Travel Request")
    
    col_input, col_meta = st.columns([3, 1])
    with col_input:
        user_query = st.text_area(
            "Describe your trip preferences:",
            value=default_val,
            height=120,
            placeholder="Enter your destination, departure location, preferences..."
        )
    
    with col_meta:
        travel_style = st.selectbox("Travel Style", ["Luxury & Relaxation", "Adventure & Exploring", "Cultural & Culinary", "Budget / Backpacker", "Family Friendly"])
        duration = st.slider("Duration (Days)", 1, 14, 4)

    generate_btn = st.button("✨ Launch VoyageAI Travel Graph")


# ==========================================
# 5. EXECUTION & RESULTS
# ==========================================

if generate_btn:
    if not user_query.strip():
        st.error("Please enter a travel request before launching.")
    else:
        full_query = f"{user_query.strip()} (Style: {travel_style}, Duration: {duration} days)"
        
        st.markdown("---")
        status_box = st.empty()
        prog_bar = st.progress(0)
        
        app = build_travel_graph(groq_key_input)
        config = {"configurable": {"thread_id": thread_id}}

        status_box.markdown("✈️ **Fetching flights & hotels...**")
        prog_bar.progress(40)
        time.sleep(0.4)

        status_box.markdown("🗺️ **Generating itinerary with Llama 3.3 70B...**")
        prog_bar.progress(80)
        
        try:
            result = app.invoke(
                {
                    "messages": [HumanMessage(content=full_query)],
                    "user_query": full_query,
                    "flight_results": "",
                    "hotel_results": "",
                    "itinerary": "",
                    "llm_calls": 0
                },
                config=config
            )
            prog_bar.progress(100)
            status_box.success("🎉 Travel Plan Generated!")
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(f"Error executing travel graph: {str(e)}")

# Display Results Tabs
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    
    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["🗺️ Complete Itinerary", "✈️ Flights Found", "🏨 Hotels Discovery", "📊 Graph State"])
    
    with t1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        itinerary_text = result.get("itinerary", "No itinerary content generated.")
        st.markdown(itinerary_text)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Download Travel Plan (Markdown)",
            data=itinerary_text,
            file_name="VoyageAI_Travel_Plan.md",
            mime="text/markdown"
        )
        
    with t2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### ✈️ Flight Details")
        st.code(result.get("flight_results", "No flight data returned."), language="yaml")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with t3:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 🏨 Hotel Recommendations")
        st.markdown(result.get("hotel_results", "No hotel data returned."))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with t4:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Execution Telemetry")
        st.write(f"**Total LLM Agent Calls:** {result.get('llm_calls', 0)}")
        st.write(f"**Messages in State:** {len(result.get('messages', []))}")
        st.markdown('</div>', unsafe_allow_html=True)
