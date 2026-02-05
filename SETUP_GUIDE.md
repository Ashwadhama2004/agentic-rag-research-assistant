# Quick Setup Guide - No OpenAI Required

## Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Clone repository
git clone <your-repo-url>
cd multi-tenant-agentic-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Choose Your LLM Provider

#### Option A: Groq (Recommended - Fast & Free)

1. **Sign up:** https://console.groq.com
2. **Get API key:** Click "Create API Key"
3. **Add to .env:**
   ```bash
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_key_here
   LLM_MODEL=llama-3.1-8b-instant
   ```

**That's it!** No credit card needed, 6000 requests/day free.

#### Option B: Ollama (Completely Local)

1. **Install Ollama:** https://ollama.ai/download
2. **Download model:**
   ```bash
   ollama pull llama3.1
   ```
3. **Start server:**
   ```bash
   ollama serve
   ```
4. **Add to .env:**
   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   LLM_MODEL=llama3.1
   ```

### Step 3: Configure Environment

Create `.env` file:

```bash
# Copy template
cp .env.example .env

# Edit with your settings
nano .env  # or use any text editor
```

Minimum required:
```bash
# LLM (choose one from Step 2)
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here

# Embedding (automatic download)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Database
DATABASE_URL=sqlite:///./data/app.db
SECRET_KEY=change-this-to-random-string

# Storage
ENDEE_PATH=./endee_db
```

### Step 4: Initialize Database

```bash
python scripts/init_db.py
```

### Step 5: Run Application

```bash
streamlit run streamlit_app.py
```

Open browser: http://localhost:8501

---

## Troubleshooting

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "Groq API key not found"
- Check `.env` file exists
- Check `GROQ_API_KEY` is set correctly
- Restart Streamlit app

### Ollama connection error
```bash
# Check if Ollama is running
ollama list

# If not, start it
ollama serve
```

### Slow embedding generation
- First run downloads model (~80MB)
- Subsequent runs use cached model
- Use GPU if available:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

---

## Model Recommendations

### For Quick Testing (Fast)
- Embedding: `all-MiniLM-L6-v2`
- LLM: Groq `llama-3.1-8b-instant`

### For Best Quality
- Embedding: `all-mpnet-base-v2`
- LLM: Groq `llama-3.1-70b-versatile`

### For Privacy (Offline)
- Embedding: `all-MiniLM-L6-v2`
- LLM: Ollama `llama3.1`

---

## Cost Breakdown

| Component | Cost |
|-----------|------|
| Embeddings (SentenceTransformers) | $0 |
| LLM (Groq free tier) | $0 |
| Database (SQLite) | $0 |
| Vector DB (Endee) | $0 |
| Deployment (Streamlit Cloud) | $0 |
| **Total** | **$0/month** |

---

## Next Steps

1. Upload a document via the UI
2. Ask questions about it
3. Try the agent for research tasks
4. Check metrics dashboard
5. Deploy to Streamlit Cloud (optional)

---

## Deployment to Streamlit Cloud

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect repository
4. Add secrets in dashboard:
   ```toml
   GROQ_API_KEY = "gsk_..."
   LLM_PROVIDER = "groq"
   LLM_MODEL = "llama-3.1-70b-versatile"
   EMBEDDING_MODEL = "all-MiniLM-L6-v2"
   DATABASE_URL = "sqlite:///./app.db"
   SECRET_KEY = "your-secret-key"
   ENDEE_PATH = "./endee_db"
   ```
5. Deploy!

Your app will be live at: `https://your-app-name.streamlit.app`

---

## Support

For issues, check:
- Groq Console: https://console.groq.com
- Ollama Docs: https://ollama.ai/docs
- SentenceTransformers: https://www.sbert.net/

Happy building! 🚀
