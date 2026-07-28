# ✈️ VoyageAI — Autonomous Travel Planner UI

A luxury, glassmorphism-styled Streamlit Web Application for the **LangGraph Autonomous Travel Planner**.

![VoyageAI Banner](assets/travel_banner.png)

---

## 🌟 Key Features

- ✈️ **Flight Agent**: Integration with AviationStack for real-time flight search.
- 🏨 **Hotel Agent**: Integration with Tavily AI for boutique stay discoveries.
- 🗺️ **Itinerary Agent**: Powered by **Llama 3.3 70B Versatile** via Groq API.
- ⚡ **LangGraph Workflow**: StateGraph stateful orchestration with PostgreSQL memory fallback.
- 🎨 **Glassmorphism UI**: High-end dark theme, dynamic metric badges, live progress visualizer, tabbed results, and markdown download options.

---

## 📁 Repository Structure

```
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
├── assets/
│   ├── travel_banner.png    # Hero UI banner image
│   └── travel_card.png      # Feature graphics
├── tools/
│   ├── __init__.py
│   ├── tavily_tool.py       # Tavily AI Search Tool
│   └── flight_tool.py       # AviationStack Flight Search Tool
├── app.py                   # Main Streamlit Application
├── requirements.txt         # Dependencies for Streamlit Cloud
├── .env.example             # Template for API keys
└── .gitignore               # Keeps API keys safe
```

---

## 🚀 How to Run Locally

1. **Clone or Open Project Directory:**
   ```bash
   cd streamlit_project
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   AVIATIONSTACK_API_KEY=your_aviationstack_api_key
   DATABASE_URL=postgresql://postgres:password@localhost:5432/langgraph_memory_demo
   ```

4. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ How to Deploy on Streamlit Community Cloud

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Deploy VoyageAI Streamlit App"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Deploy on Streamlit:**
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Click **"New App"** -> Select your repository & branch (`main`).
   - Set Main file path: `app.py`.
   - In **Advanced Settings -> Secrets**, paste your API keys:
     ```toml
     GROQ_API_KEY = "your_groq_key"
     TAVILY_API_KEY = "your_tavily_key"
     AVIATIONSTACK_API_KEY = "your_aviation_key"
     ```
   - Click **Deploy**! 🚀
