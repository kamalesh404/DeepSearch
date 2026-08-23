<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0066CC,100:00CED1&height=200&section=header&text=DeepSearch&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=AI-Powered+Search+Engine+with+Knowledge+Graphs&descAlignY=58&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-0.1-1C3C3C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Neo4j-5.0-008CC1?style=for-the-badge&logo=neo4j&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector-FF4500?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/kamalesh404/DeepSearch?style=social" />
  <img src="https://img.shields.io/github/forks/kamalesh404/DeepSearch?style=social" />
  <img src="https://img.shields.io/github/license/kamalesh404/DeepSearch" />
</p>

---

## What is DeepSearch?

DeepSearch is an **AI-powered search engine** that goes beyond keyword matching. It builds knowledge graphs from indexed content, uses RAG (Retrieval-Augmented Generation) to provide AI-synthesized answers, and lets you explore connections between concepts visually.

### Key Features

- **Semantic Search** — Understand meaning, not just keywords
- **Knowledge Graphs** — Visualize relationships between concepts
- **RAG-Powered Answers** — AI-synthesized responses with source citations
- **Multi-Source Indexing** — Crawl websites, parse PDFs, DOCX, and TXT files
- **Real-Time Indexing** — Add URLs or documents and search instantly
- **Graph Exploration** — Navigate knowledge graphs interactively

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DeepSearch                         │
├──────────────┬──────────────┬───────────────────────┤
│   Frontend   │   Backend    │   Storage Layer       │
│   React 18   │   FastAPI    │   ChromaDB (vectors)  │
│   Vite       │   Python 3.11│   Neo4j (graph)       │
│   Tailwind   │   LangChain  │   PostgreSQL (meta)   │
│   D3.js      │   OpenAI     │                       │
├──────────────┴──────────────┴───────────────────────┤
│              Search Pipeline                         │
│   Query → Embed → Vector Search → Rank → RAG Answer │
├─────────────────────────────────────────────────────┤
│              Indexing Pipeline                       │
│   URL/Doc → Parse → Chunk → Embed → Store           │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 18, Vite, Tailwind CSS | UI/UX |
| Visualization | D3.js, React Flow | Knowledge graph rendering |
| Backend | FastAPI, Python 3.11 | API server |
| AI | LangChain, OpenAI GPT-4 | RAG & embeddings |
| Vector DB | ChromaDB | Semantic search |
| Graph DB | Neo5.x | Knowledge graphs |
| Crawling | BeautifulSoup, httpx | Web scraping |
| CI/CD | GitHub Actions | Automated testing |

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- OpenAI API key

### Quick Start (Docker)
```bash
git clone https://github.com/kamalesh404/DeepSearch.git
cd DeepSearch
cp .env.example .env  # Add your OpenAI API key
docker-compose up --build
```

### Manual Setup
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

---

## API Documentation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/search` | Semantic search with AI answers |
| POST | `/api/index/url` | Index a webpage |
| POST | `/api/index/document` | Upload & index a document |
| GET | `/api/graph/explore` | Explore knowledge graph |
| GET | `/api/stats` | Index statistics |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00CED1,100:0066CC&height=120&section=footer" width="100%"/>
</p>
