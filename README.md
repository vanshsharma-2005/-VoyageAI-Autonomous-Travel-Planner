# ✈ Voyager — AI Travel Planner

An AI travel-planning agent with a Streamlit chat interface. Ask about a
trip in plain language and Voyager pulls live flight data, searches for
hotels, and hands back a full itinerary — powered by a LangGraph multi-agent
pipeline running on Groq's Llama 3.3 70B.

## How it works

Every message runs through a fixed 4-step pipeline:

```
   your message
        │
        ▼
 ┌───────────────┐
 │ flight_agent  │  → AviationStack flight search
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │  hotel_agent  │  → Tavily web search for hotels
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │itinerary_agent│  → Groq LLM drafts a day-by-day plan
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │  final_agent  │  → Groq LLM writes the final response
 └───────┬───────┘
         ▼
    reply in chat
```

Conversation memory is handled by a LangGraph checkpointer, keyed per
browser session. If `DATABASE_URL` is configured it persists to Postgres;
otherwise it falls back to in-memory (session-only) automatically.

## Tech stack

| Layer            | Tool                                   |
|-------------------|-----------------------------------------|
| UI                | Streamlit                                |
| Agent orchestration | LangGraph (`StateGraph`)               |
| LLM               | Groq — Llama 3.3 70B Versatile           |
| Flight data       | [AviationStack](https://aviationstack.com/) |
| Web / hotel search | [Tavily](https://www.tavily.com/)       |
| Conversation memory | Postgres (optional) / in-memory fallback |

## Project structure

```
travel_planner/
├── frontend.py           # Streamlit UI + LangGraph pipeline (run this)
├── main.ipynb             # original notebook prototype
├── tools/
│   ├── flight_tool.py      # AviationStack flight search
│   └── tavily_tool.py      # Tavily web search
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # fill in your real API keys
streamlit run frontend.py
```

## Environment variables

| Variable                 | Required | Notes                                              |
|---------------------------|----------|-----------------------------------------------------|
| `GROQ_API_KEY`             | Yes      | [console.groq.com](https://console.groq.com/)       |
| `TAVILY_API_KEY`           | Yes      | [tavily.com](https://www.tavily.com/)                |
| `AVIATIONSTACK_API_KEY`    | Yes      | [aviationstack.com](https://aviationstack.com/)       |
| `DATABASE_URL`             | No       | Postgres connection string; omit for in-memory-only memory |



## Known limitations

- The pipeline is fixed, not intent-routed: it always fetches both flights
  and hotels, even for questions unrelated to either.
- `search_flights` doesn't filter by the query text — it currently returns
  AviationStack's default 5 flights regardless of route asked.
- Without `DATABASE_URL`, conversation memory resets whenever the app
  restarts or redeploys.
