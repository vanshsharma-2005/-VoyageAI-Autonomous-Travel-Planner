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

# Custom Tools
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

# Load environment variables
load_dotenv()

# ==========================================
# 1. PAGE SETUP & GLASSMORPHISM DESIGN SYSTEM
# ==========================================
st.set_page_config(
    page_title="VoyageAI — Luxury Travel Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #090d16 90%);
        color: #f8fafc;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }

    /* Metric Cards */
    .metric-badge {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-badge h4 {
        margin: 0;
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-badge p {
        margin: 4px 0 0 0;
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Hero Typography */
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Glowing Primary Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 30px rgba(217, 70, 239, 0.6) !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: rgba(30, 41, 59, 0.4);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        font-weight: 600;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(139, 92, 246, 0.3)) !important;
        border: 1px solid rgba(139, 92, 246, 0.5) !important;
        color: #ffffff !important;
    }

    /* Step Timeline Pills */
    .agent-step {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 8px;
    }
    .agent-step.active {
        border-color: #38bdf8;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
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
    # Fallback to env key if not passed directly
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
            "messages": [AIMessage(content="✈️ Flight information retrieved successfully.")],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    def hotel_agent(state: TravelState):
        query = f"Best luxury and boutique hotels for {state['user_query']}"
        hotel_results = tavily_search(query)
        return {
            "hotel_results": hotel_results,
            "messages": [AIMessage(content="🏨 Hotel details fetched successfully.")],
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

        Please structure the response clearly with formatting, emojis, daily schedules, estimated costs, and expert tips.
        """
        response = llm.invoke([
            SystemMessage(content="You are an elite, world-class luxury travel agent."),
            HumanMessage(content=prompt)
        ])
        return {
            "itinerary": response.content,
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    def final_agent(state: TravelState):
        final_prompt = f"""
        Provide a polished executive travel summary consolidating everything:

        Flights:
        {state['flight_results']}

        Hotels:
        {state['hotel_results']}

        Detailed Itinerary:
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

    # Checkpointer Setup (Postgres with MemorySaver fallback)
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
# 3. SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### ✈️ VoyageAI Control Center")
    st.caption("Powered by LangGraph & Groq Llama 3.3 70B")
    
    st.markdown("---")
    groq_key_input = st.text_input(
        "🔑 Groq API Key", 
        type="password", 
        value=os.getenv("GROQ_API_KEY", ""), 
        help="Defaults to your environment key if left empty."
    )
    
    thread_id = st.text_input("🧵 Session Thread ID", value="user_vansh", help="Persists graph state memory.")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Diagnostics")
    st.markdown(f"**Database Memory:** `{st.session_state.get('db_status', 'Initializing...')}`")
    st.markdown(f"**LangGraph State:** `Active Graph Ready`")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Sample Trips")
    if st.button("🇯🇵 5 Days in Tokyo ($2,500 Budget)"):
        st.session_state["sample_prompt"] = "5 days in Tokyo Japan, luxury food & tech culture, $2500 budget, starting from New York"
    if st.button("🇫🇷 Romantic Weekend in Paris"):
        st.session_state["sample_prompt"] = "3 day romantic weekend in Paris France, 5-star hotel near Eiffel Tower, fine dining & museums"
    if st.button("🇨🇭 Swiss Alps Alpine Retreat"):
        st.session_state["sample_prompt"] = "7 days scenic train & hiking trip in Swiss Alps, Zurich to Zermatt, cozy luxury chalets"


# ==========================================
# 4. MAIN CONTENT AREA
# ==========================================

# Hero Banner & Title
hero_banner_path = os.path.join(os.path.dirname(__file__), "assets", "travel_banner.png")
if os.path.exists(hero_banner_path):
    st.image(hero_banner_path, use_column_width=True)

st.markdown('<h1 class="hero-title">VoyageAI Agentic Travel Planner</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Multi-Agent AI Workflow orchestrating live flight data, hotel discovery, and personalized itineraries.</p>', unsafe_allow_html=True)

# Top Metric Summary Cards
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

# Input Section inside a Glass Container
default_val = st.session_state.get("sample_prompt", "Plan a 4-day trip to Bali from London with luxury resort recommendations and adventure activities.")

with st.container():
    st.markdown("### 🧳 Customize Your Travel Request")
    
    col_input, col_meta = st.columns([3, 1])
    with col_input:
        user_query = st.text_area(
            "Describe your dream trip details:",
            value=default_val,
            height=120,
            placeholder="e.g. 5 days in Rome with wine tasting and boutique hotels..."
        )
    
    with col_meta:
        travel_style = st.selectbox("Travel Style", ["Luxury & Relaxation", "Adventure & Exploring", "Cultural & Culinary", "Budget / Backpacker", "Family Friendly"])
        duration = st.slider("Duration (Days)", 1, 14, 4)

    generate_btn = st.button("✨ Launch VoyageAI Travel Graph")

# ==========================================
# 5. GRAPH EXECUTION & OUTPUT RENDERING
# ==========================================

if generate_btn:
    if not user_query.strip():
        st.error("Please enter a travel request before launching.")
    else:
        full_query = f"{user_query.strip()} (Style: {travel_style}, Duration: {duration} days)"
        
        # Execution visual container
        st.markdown("---")
        st.markdown("### ⚙️ Live LangGraph Workflow Engine")
        
        status_box = st.empty()
        prog_bar = st.progress(0)
        
        # Build app instance
        app = build_travel_graph(groq_key_input)
        
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # Step 1: Flights
        status_box.markdown("✈️ **[Node 1/4] Flight Agent:** Querying flight data via AviationStack...")
        prog_bar.progress(25)
        time.sleep(0.5)

        # Step 2: Hotels
        status_box.markdown("🏨 **[Node 2/4] Hotel Agent:** Searching luxury stay options via Tavily Search...")
        prog_bar.progress(50)
        time.sleep(0.5)

        # Step 3 & 4: Invoke full graph
        status_box.markdown("🗺️ **[Node 3 & 4/4] Itinerary & Final Agent:** Reasoning with Llama 3.3 70B...")
        prog_bar.progress(85)
        
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
            status_box.success("🎉 Travel Plan Successfully Generated!")
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(f"Error executing travel graph: {str(e)}")

# Display Results if available
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    
    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["🗺️ Complete Itinerary", "✈️ Flights Found", "🏨 Hotels Discovery", "📊 Graph State & Metrics"])
    
    with t1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        itinerary_text = result.get("itinerary", "No itinerary content generated.")
        st.markdown(itinerary_text)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download Markdown button
        st.download_button(
            label="📥 Download Travel Plan (Markdown)",
            data=itinerary_text,
            file_name="VoyageAI_Travel_Plan.md",
            mime="text/markdown"
        )
        
    with t2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ✈️ Flight Search Results")
        st.code(result.get("flight_results", "No flight data returned."), language="yaml")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with t3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏨 Hotel & Accommodation Discovery")
        st.markdown(result.get("hotel_results", "No hotel data returned."))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with t4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Execution Telemetry")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total LLM Agent Calls", result.get("llm_calls", 0))
        with c2:
            st.metric("Messages In State", len(result.get("messages", [])))
            
        st.markdown("#### Conversation History Log")
        for i, msg in enumerate(result.get("messages", [])):
            st.text(f"[{i+1}] {msg.__class__.__name__}: {msg.content[:150]}...")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("VoyageAI • Autonomous Agentic Travel Intelligence System • Ready for Streamlit Cloud Deployment")
