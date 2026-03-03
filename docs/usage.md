# OpenClaw Memory — Code Wiki

> 基於原始碼完整分析產生，非抄寫既有文件

---

## 目錄

1. [系統總覽](#系統總覽)
2. [目錄結構](#目錄結構)
3. [架構圖](#架構圖)
4. [三層記憶模型](#三層記憶模型)
5. [兩條主要管線](#兩條主要管線)
6. [模組詳解](#模組詳解)
7. [資料庫 Schema](#資料庫-schema)
8. [API 端點一覽](#api-端點一覽)
9. [使用指南：如何跑起來](#使用指南如何跑起來)
10. [使用指南：常見操作](#使用指南常見操作)
11. [使用指南：LongMemEval Benchmark Service](#使用指南longmemeval-benchmark-service)
12. [設定參數速查](#設定參數速查)

---

## 系統總覽

OpenClaw Memory 是一個 **三層記憶系統**，設計給 chatbot 使用。核心概念：

- **L1 Working Memory** — 當前對話的短期記憶（session 結束即清除）
- **L2 Episodic Memory** — 時間衰減的事件記憶（會隨時間淡忘）
- **L3 Semantic Memory** — 永久性知識（事實、偏好、決策，不會衰減）

系統提供兩條管線：
- **Ingest Pipeline** — 對話 → 結構化記憶（寫入路徑）
- **Retrieval Pipeline** — 查詢 → 分層搜尋 → 排序 → 答案（讀取路徑）

技術棧：FastAPI + PostgreSQL (pgvector) + 多種 Embedding Provider + LLM

---

## 目錄結構

```
src/openclaw_memory/
├── __init__.py              # 套件入口，匯出公開 API
├── app.py                   # FastAPI 應用工廠 (create_app)
├── config.py                # Pydantic 設定，所有 OPENCLAW_ 環境變數
├── dependencies.py          # FastAPI 依賴注入（單例管理）
│
├── core/                    # ✦ 核心抽象層
│   ├── embeddings.py        #   Embedding 提供者協定 + 工廠
│   ├── service.py           #   精簡版 MemoryService
│   └── types.py             #   核心型別（MemoryIndex, MemoryContext, ExtractedMemory...）
│
├── models/                  # ✦ Pydantic 請求/回應模型
│   ├── memory.py            #   記憶 CRUD 模型
│   ├── search.py            #   搜尋三層模型 + RAG 模型
│   └── session.py           #   Session 生命週期模型
│
├── db/                      # ✦ PostgreSQL 資料庫層
│   ├── connection.py        #   連線池管理
│   ├── schema.py            #   Schema 遷移（idempotent DDL）
│   └── queries.py           #   SQL 查詢函式（向量/關鍵字/CRUD）
│
├── memory/                  # ✦ 三層記憶實作
│   ├── working.py           #   L1 WorkingMemory（PG-backed）
│   ├── episodic.py          #   L2 store_episodic / store_session_episode
│   └── semantic.py          #   L3 store_semantic + table_for_memory
│
├── pipeline/                # ✦ 管線（核心業務邏輯）
│   ├── ingest/
│   │   ├── sensory.py       #   SensoryConfig + prepare_for_extraction
│   │   ├── extraction.py    #   extract_memories（LLM 萃取）
│   │   ├── conflict.py      #   resolve_conflict + apply_resolution
│   │   └── normalize.py     #   正規化工具
│   └── retrieval/
│       ├── search.py        #   三層搜尋（compact/timeline/detail/full）
│       ├── hybrid.py        #   BM25 + Vector 混合
│       ├── ranking.py       #   時間衰減 + MMR + LLM rerank
│       └── answer.py        #   RAG 答案生成 + 引用
│
├── consolidation/           # ✦ 離線合併引擎
│   ├── consolidator.py      #   MemoryConsolidator（叢集/合併/腐化檢測）
│   ├── dedup.py             #   Embedding 去重
│   └── promotion.py         #   Episodic → Semantic 晉升
│
├── routers/                 # ✦ FastAPI 路由
│   ├── health.py            #   GET /health
│   ├── session.py           #   POST /session/{start,message,end}
│   ├── memory.py            #   POST /memory/add, GET/DELETE /memory/{user_id}
│   └── search.py            #   POST /search, /search/detail, /search/answer, GET /search/timeline
│
└── utils/                   # ✦ 共用工具
    ├── similarity.py        #   cosine / jaccard 相似度
    ├── text.py              #   文字處理（JSON 解析/截斷/正規化）
    └── tokens.py            #   Token 估算（len // 4）
```

---

## 架構圖

### 整體系統架構

```mermaid
graph TB
    subgraph "入口點"
        API["FastAPI Server<br/>(app.py)"]
        CLI["CLI Tool<br/>(ocmem)"]
    end

    subgraph "路由層"
        RH["GET /health"]
        RS["POST /session/*"]
        RM["POST /memory/*"]
        RSE["POST /search/*"]
    end

    subgraph "服務層"
        SVC["MemoryService<br/>(core/service.py)"]
        DEP["dependencies.py<br/>單例管理"]
    end

    subgraph "寫入管線 (Ingest)"
        SENS["Sensory Filter<br/>感知過濾"]
        EXT["Extraction<br/>LLM 記憶萃取"]
        CONF["Conflict Resolution<br/>衝突解析"]
        CONS["Consolidation<br/>離線合併"]
    end

    subgraph "讀取管線 (Retrieval)"
        SRCH["Hybrid Search<br/>向量 + 關鍵字"]
        RANK["Ranking Pipeline<br/>衰減 + MMR + LLM"]
        ANS["Answer Generation<br/>RAG 答案"]
    end

    subgraph "記憶層"
        L1["L1 Working Memory<br/>（session 範圍）"]
        L2["L2 Episodic Memory<br/>（時間衰減）"]
        L3["L3 Semantic Memory<br/>（永久儲存）"]
        CAN["Canonical Memories<br/>（合併後事實）"]
    end

    subgraph "儲存層"
        PG["PostgreSQL<br/>+ pgvector"]
        SQLITE["SQLite<br/>+ FTS5（本地索引）"]
        REDIS["Redis<br/>（可選 L1 快取）"]
    end

    subgraph "外部服務"
        EMB["Embedding Provider<br/>OpenAI / Gemini / Voyage / Local"]
        LLM["LLM<br/>gpt-4o-mini"]
    end

    API --> RH & RS & RM & RSE
    RS & RM & RSE --> DEP --> SVC
    CLI --> SVC

    SVC --> SENS --> EXT --> CONF
    CONF --> L2 & L3 & CAN
    CONS --> CAN

    SVC --> SRCH --> RANK --> ANS
    SRCH --> L1 & L2 & L3 & CAN

    L1 --> PG & REDIS
    L2 --> PG
    L3 --> PG
    CAN --> PG

    EXT --> LLM
    CONF --> LLM
    RANK --> LLM
    ANS --> LLM
    SRCH --> EMB
    CONF --> EMB
```

### 寫入管線（Ingest Pipeline）

```mermaid
flowchart LR
    A["對話訊息<br/>(messages)"] --> B["Sensory Filter<br/>過濾低資訊<br/>壓縮長文<br/>主題分段"]
    B --> C["LLM Extraction<br/>萃取記憶<br/>分類(fact/pref/event/decision)<br/>評信心值"]
    C --> D{"信心 >= 閾值?"}
    D -- 否 --> SKIP["丟棄"]
    D -- 是 --> E["Conflict Resolution"]
    E --> F{"比對結果"}
    F -- "新記憶" --> ADD["ADD → 插入新列"]
    F -- "更新既有" --> UPD["UPDATE → 修改既有"]
    F -- "取代" --> SUP["SUPERSEDE → 停用舊、插入新"]
    F -- "刪除" --> DEL["DELETE → 軟刪除"]
    F -- "無需動作" --> NOOP["NOOP → 跳過"]
```

### 讀取管線（Retrieval Pipeline）

```mermaid
flowchart LR
    Q["使用者查詢"] --> VEC["向量搜尋<br/>(pgvector cosine)"]
    Q --> KW["關鍵字搜尋<br/>(tsvector BM25)"]
    VEC --> MIX["Hybrid Merge<br/>加權合併<br/>vec:0.7 + text:0.3"]
    KW --> MIX
    MIX --> TD["Temporal Decay<br/>指數衰減<br/>半衰期 30 天"]
    TD --> MMR["MMR<br/>多樣性重排<br/>λ=0.7"]
    MMR --> RR["LLM Rerank<br/>（可選）"]
    RR --> TOP["Top-K 結果"]
    TOP --> ANS["RAG Answer<br/>結構化答案<br/>附引用"]
```

### 三層搜尋（Progressive Disclosure）

```mermaid
flowchart TB
    subgraph "Tier 1 — Compact（~50 tokens/result）"
        T1["POST /search<br/>回傳 id, title, type, score"]
    end
    subgraph "Tier 2 — Timeline（~200 tokens/result）"
        T2["GET /search/timeline/{id}<br/>回傳 content + neighbors"]
    end
    subgraph "Tier 3 — Detail（完整內容）"
        T3["POST /search/detail<br/>回傳 full content + metadata"]
    end

    T1 -- "選擇感興趣的 ID" --> T2
    T2 -- "選擇需要完整內容的 ID" --> T3
```

### Session 生命週期

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    participant L1 as L1 Working
    participant L2 as L2 Episodic
    participant L3 as L3 Semantic

    C->>S: POST /session/start {user_id, session_id}
    S->>L1: 初始化 session，清除過期 L1

    loop 對話進行中
        C->>S: POST /session/message {role, content}
        S->>L1: append(role, content)
    end

    C->>S: POST /session/end {user_id, session_id}
    S->>L1: 讀取所有 L1 訊息
    S->>S: Sensory → Extract → Conflict Resolve
    S->>L2: store_episodic（事件類記憶）
    S->>L3: store_semantic（事實/偏好/決策）
    S->>L1: clear_session
    S-->>C: {extracted: N, stored: M}
```

### 離線合併（Consolidation）

```mermaid
flowchart TB
    START["啟動合併<br/>(sleep-time)"] --> LOAD["載入所有 active<br/>canonical memories"]
    LOAD --> CLUSTER["Embedding 叢集<br/>(cosine >= 0.90)"]
    CLUSTER --> MERGE["LLM 合併叢集<br/>→ 標記舊為 superseded<br/>→ 插入合併結果"]
    MERGE --> STALE["腐化檢測<br/>(confidence < 0.7<br/>+ 超過 180 天)"]
    STALE --> ACT{"LLM 評估"}
    ACT -- "keep" --> DONE
    ACT -- "delete" --> SOFT["軟刪除"]
    ACT -- "abstract" --> ABS["抽象化<br/>→ 新 fact 記憶"]
    SOFT --> DONE
    ABS --> DONE
    DONE --> PROMO["Episodic 晉升<br/>(event > 90 天<br/>→ fact/preference)"]
    PROMO --> FIN["完成<br/>回報 ConsolidationReport"]
```

---

## 三層記憶模型

| 層級 | 名稱 | 資料表 | 特性 | 用途 |
|------|------|--------|------|------|
| **L1** | Working Memory | `working_messages` | Session 範圍，最多 20 筆，用完即清 | 當前對話上下文 |
| **L2** | Episodic Memory | `episodic_memories` | 時間衰減（半衰期 30 天），event 類型為主 | 近期事件記錄 |
| **L3** | Semantic Memory | `semantic_memories` + `canonical_memories` | 永久儲存，不衰減 | 長期事實、偏好、決策 |

### 記憶類型（memory_type）

| 類型 | 說明 | 典型去向 |
|------|------|----------|
| `preference` | 使用者偏好 | L3 semantic |
| `fact` | 事實知識 | L3 semantic |
| `decision` | 決策記錄 | L3 semantic |
| `event` | 事件記錄 | L2 episodic |
| `session` | 整段對話摘要 | L2 episodic |

---

## 兩條主要管線

### Ingest Pipeline（寫入路徑）

1. **Sensory Filter** (`pipeline/ingest/sensory.py`)
   - 過濾低資訊訊息（ok, sure, thanks 等）
   - 壓縮長文（TF-IDF 句子評分）
   - 主題分段（Jaccard 相似度 + 關鍵字偵測）
   - 裁剪到 token 預算

2. **Extraction** (`pipeline/ingest/extraction.py`)
   - LLM 從對話中萃取結構化記憶
   - 產出：content, memory_type, confidence, memory_key, value, event_time
   - 過濾低信心記憶（`extraction_min_confidence` 閾值）

3. **Conflict Resolution** (`pipeline/ingest/conflict.py`)
   - 兩種模式：規則式 / LLM CRUD 式
   - 比對候選記憶 vs 既有 canonical memories
   - 五種動作：ADD, UPDATE, SUPERSEDE, DELETE, NOOP

4. **Consolidation** (`consolidation/consolidator.py`)
   - 離線執行（sleep-time）
   - 叢集近似記憶 → LLM 合併
   - 腐化檢測 → 刪除或抽象化
   - Episodic → Semantic 晉升

### Retrieval Pipeline（讀取路徑）

1. **Search** (`pipeline/retrieval/search.py`)
   - 向量搜尋：pgvector cosine similarity
   - 關鍵字搜尋：PostgreSQL tsvector + BM25
   - 混合合併：加權 (預設 vec:0.7 + text:0.3)
   - 跨所有三層搜尋

2. **Ranking** (`pipeline/retrieval/ranking.py`)
   - 時間衰減：指數半衰期（預設 30 天）
   - MMR：多樣性重排（λ=0.7）
   - LLM Rerank：可選的精確度提升

3. **Answer** (`pipeline/retrieval/answer.py`)
   - RAG 答案生成
   - 結構化輸出：answer, confidence, evidence[], abstain
   - 自動重試（若 LLM 棄權但有 evidence）

---

## 模組詳解

### `core/service.py` — MemoryService（新版精簡）

主要協調器，所有高層操作的入口點。

```python
class MemoryService:
    def __init__(self, embedding_provider, settings, llm_fn=None)

    # Session 生命週期
    def start_session(self, conn, user_id, session_id)
    def record_message(self, conn, user_id, session_id, role, content)
    def end_session(self, conn, user_id, session_id) -> dict

    # 寫入
    def ingest_conversation(self, conn, user_id, conversation, *, session_id=None) -> dict
    def add_memory(self, conn, user_id, content, *, memory_type="fact", metadata=None) -> str

    # 搜尋
    def search(self, conn, user_id, query, *, top_k=None, include_working=False, session_id=None) -> list[MemorySearchResult]
    def search_compact(self, conn, user_id, query, *, limit=10) -> list[MemoryIndex]
    def search_timeline(self, conn, user_id, memory_id, *, depth_before=3, depth_after=3) -> MemoryContext | None
    def search_detail(self, conn, memory_ids) -> list[MemorySearchResult]

    # RAG
    def answer(self, conn, user_id, query, *, top_k=6, session_id=None) -> AnswerPayload

    # CRUD
    def get_user_memories(self, conn, user_id, *, limit=100, offset=0) -> list
    def delete_memory(self, conn, memory_id, user_id) -> bool
```

### `core/embeddings.py` — Embedding Provider

```python
class EmbeddingProvider(Protocol):
    id: str          # "openai" | "gemini" | "voyage" | "local"
    model: str       # 模型名稱
    def embed_query(text: str) -> list[float]
    def embed_batch(texts: list[str]) -> list[list[float]]

# 工廠函式
def create_embedding_provider(
    provider: str = "auto",  # auto | openai | gemini | voyage | local
    model: str | None = None,
    api_key: str | None = None,
) -> EmbeddingProvider
```

**支援的 Provider：**

| Provider | 預設模型 | 環境變數 |
|----------|---------|----------|
| `openai` | text-embedding-3-small | `OPENAI_API_KEY` |
| `gemini` | gemini-embedding-001 | `GEMINI_API_KEY` 或 `GOOGLE_API_KEY` |
| `voyage` | voyage-4-large | `VOYAGE_API_KEY` |
| `local` | llama-cpp GGUF | 需要 `llama-cpp-python` |

### `memory/working.py` — L1 Working Memory

```python
class WorkingMemory:
    def __init__(max_messages=20)
    def append(conn, user_id, session_id, role, content)
    def get_recent(conn, user_id, session_id, limit=None) -> list[dict]
    def clear_session(conn, user_id, session_id)
    def clear_user(conn, user_id)
    def to_search_results(messages, user_id, session_id) -> list[dict]
```

### `consolidation/consolidator.py` — 離線合併引擎

```python
class MemoryConsolidator:
    def __init__(embedding_provider, llm_fn, similarity_threshold=0.90, max_cluster_size=10)
    def consolidate(conn, user_id) -> ConsolidationReport
```

---

## 資料庫 Schema

### PostgreSQL 表格

```mermaid
erDiagram
    working_messages {
        uuid id PK
        text user_id
        text session_id
        text role
        text content
        timestamptz created_at
    }

    episodic_memories {
        uuid id PK
        text user_id
        text session_id
        text content
        vector_1536 embedding
        text memory_type
        timestamptz created_at
        jsonb metadata
        tsvector tsv "auto-generated"
    }

    semantic_memories {
        uuid id PK
        text user_id
        text content
        vector_1536 embedding
        text memory_type
        timestamptz created_at
        jsonb metadata
        tsvector tsv "auto-generated"
    }

    canonical_memories {
        uuid id PK
        text user_id
        text memory_key "UNIQUE with user_id where active"
        text value
        text memory_type
        float confidence
        text event_time
        text status "active|superseded|deleted"
        uuid supersedes_memory_id FK
        vector_1536 embedding
        timestamptz created_at
        timestamptz updated_at
        jsonb metadata
        uuid_array consolidated_from
        int consolidation_round
        timestamptz last_consolidated_at
    }

    short_term_buffer {
        serial id PK
        text user_id
        text session_id
        jsonb messages
        text topic_summary
        int token_count
        timestamptz created_at
        timestamptz updated_at
    }

    memory_update_queue {
        serial id PK
        text user_id
        jsonb payload
        text status
        int attempts
        text last_error
        timestamptz available_at
        timestamptz created_at
        timestamptz updated_at
    }

    user_profiles {
        text user_id PK
        jsonb profile
        timestamptz updated_at
    }

    consolidation_log {
        serial id PK
        text user_id
        timestamptz started_at
        timestamptz finished_at
        int memories_scanned
        int memories_merged
        int memories_deleted
        int memories_abstracted
        text status
        text error
        jsonb metadata
    }

    canonical_memories ||--o{ canonical_memories : "supersedes"
```

### 索引

| 表格 | 索引類型 | 用途 |
|------|---------|------|
| episodic_memories | HNSW (vector) | 向量相似度搜尋 |
| episodic_memories | GIN (tsv) | 全文搜尋 |
| episodic_memories | BTREE (user_id, created_at) | 時間範圍查詢 |
| semantic_memories | HNSW (vector) | 向量相似度搜尋 |
| semantic_memories | GIN (tsv) | 全文搜尋 |
| canonical_memories | HNSW (vector) | 向量相似度搜尋 |
| canonical_memories | UNIQUE (user_id, memory_key) WHERE active | 唯一約束 |

---

## API 端點一覽

### Health

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/health` | 健康檢查，回傳 DB 狀態 |

### Session

| Method | Path | Request | Response |
|--------|------|---------|----------|
| `POST` | `/session/start` | `{user_id, session_id}` | `{status: "started"}` |
| `POST` | `/session/message` | `{user_id, session_id, role, content}` | `{status: "recorded"}` |
| `POST` | `/session/end` | `{user_id, session_id}` | `{extracted, stored}` |

### Memory

| Method | Path | Request | Response |
|--------|------|---------|----------|
| `POST` | `/memory/add` | `{user_id, conversation?, content?, memory_type?, metadata?}` | `{stored, extracted, memory_id?}` |
| `GET` | `/memory/{user_id}?limit=100&offset=0` | — | `{user_id, memories[], total}` |
| `DELETE` | `/memory/{user_id}/{memory_id}` | — | `{deleted, memory_id}` |

### Search

| Method | Path | Request | Response | Tier |
|--------|------|---------|----------|------|
| `POST` | `/search` | `{user_id, query, limit?}` | `{results[{id, title, type, score}]}` | Tier 1 |
| `GET` | `/search/timeline/{memory_id}?user_id=&depth_before=3&depth_after=3` | — | `{id, content, neighbors[]}` | Tier 2 |
| `POST` | `/search/detail` | `{memory_ids[]}` | `{results[{id, content, metadata}]}` | Tier 3 |
| `POST` | `/search/answer` | `{user_id, query, top_k?, session_id?}` | `{answer, confidence, abstain, evidence[]}` | RAG |

---

## 使用指南：如何跑起來

### 1. 環境變數設定

```bash
# 必要：PostgreSQL 連線
export OPENCLAW_PG_DSN="postgresql://user:pass@localhost:5432/openclaw_memory"

# 必要：Embedding Provider（擇一）
export OPENAI_API_KEY="sk-..."
# 或
export GEMINI_API_KEY="..."
# 或
export VOYAGE_API_KEY="..."

# 可選：LLM 模型
export OPENCLAW_LLM_MODEL="gpt-4o-mini"  # 預設值

# 可選：Embedding 設定
export OPENCLAW_EMBEDDING_PROVIDER="openai"  # auto|openai|gemini|voyage|local
export OPENCLAW_EMBEDDING_MODEL="text-embedding-3-small"

# 可選：伺服器設定
export OPENCLAW_HOST="0.0.0.0"
export OPENCLAW_PORT=8000
export OPENCLAW_DEBUG=true
```

### 2. 安裝

```bash
pip install -e .
```

### 3. 啟動 API Server

```python
from openclaw_memory import create_app

app = create_app()
# 然後用 uvicorn 跑：
# uvicorn openclaw_memory:create_app --factory --host 0.0.0.0 --port 8000
```

或直接用命令列：

```bash
uvicorn openclaw_memory.app:create_app --factory --host 0.0.0.0 --port 8000
```

### 4. 啟動時自動做的事

1. 建立 PostgreSQL 連線池
2. 執行 Schema Migration（idempotent，安全重複執行）
3. 建立 Embedding Provider
4. 初始化 MemoryService 單例

---

## 使用指南：常見操作

### 典型對話流程（透過 API）

```bash
# 1. 開始 session
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "session_id": "sess_abc"}'

# 2. 記錄訊息（重複執行）
curl -X POST http://localhost:8000/session/message \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "session_id": "sess_abc", "role": "user", "content": "我喜歡吃拉麵"}'

curl -X POST http://localhost:8000/session/message \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "session_id": "sess_abc", "role": "assistant", "content": "好的，我記住了！"}'

# 3. 結束 session（觸發記憶萃取）
curl -X POST http://localhost:8000/session/end \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "session_id": "sess_abc"}'
# 回傳：{"extracted": 1, "stored": 1}
```

### 直接新增記憶

```bash
# 方式一：從對話萃取
curl -X POST http://localhost:8000/memory/add \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "conversation": [
      {"role": "user", "content": "我的生日是 3 月 15 日"},
      {"role": "assistant", "content": "記下來了！"}
    ]
  }'

# 方式二：直接寫入
curl -X POST http://localhost:8000/memory/add \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "content": "使用者生日是 3 月 15 日",
    "memory_type": "fact"
  }'
```

### 搜尋記憶

```bash
# Tier 1：快速索引
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "query": "生日", "limit": 10}'

# Tier 2：時間脈絡
curl "http://localhost:8000/search/timeline/MEMORY_ID?user_id=user_123"

# Tier 3：完整內容
curl -X POST http://localhost:8000/search/detail \
  -H "Content-Type: application/json" \
  -d '{"memory_ids": ["uuid-1", "uuid-2"]}'

# RAG：直接問答
curl -X POST http://localhost:8000/search/answer \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "query": "使用者的生日是什麼時候？"}'
# 回傳：{"answer": "使用者的生日是 3 月 15 日", "confidence": 0.95, "evidence": [...]}
```

### 查看和管理記憶

```bash
# 列出使用者所有記憶
curl "http://localhost:8000/memory/user_123?limit=50"

# 刪除特定記憶
curl -X DELETE "http://localhost:8000/memory/user_123/MEMORY_UUID"
```

### 直接使用 Python SDK

```python
from openclaw_memory.core.service import MemoryService
from openclaw_memory.core.embeddings import create_embedding_provider
from openclaw_memory.config import get_settings
from openclaw_memory.db.connection import get_sync_connection

settings = get_settings()
emb = create_embedding_provider("openai")
svc = MemoryService(embedding_provider=emb, settings=settings)

# 取得 DB 連線
conn = get_sync_connection(settings.pg_dsn)

# 搜尋
results = svc.search_compact(conn, user_id="user_123", query="偏好", limit=10)

# 新增記憶
memory_id = svc.add_memory(conn, user_id="user_123", content="喜歡深色主題", memory_type="preference")

# RAG 問答
answer = svc.answer(conn, user_id="user_123", query="使用者喜歡什麼主題？")
print(answer.answer, answer.confidence)
```

---

## 使用指南：LongMemEval Benchmark Service

### 目標

這一節針對 `scripts/run_longmemeval_service.sh` 的服務模式 benchmark。

已將腳本預設值調整為品質優先組合，對應以下執行需求：

- `LME_WRITE_MODE=distill`
- `LME_DISTILL_BATCH=2`
- `LME_RESOLVER_MODE=offline`
- `LME_DRAIN_MODE=after_run`
- `LME_SEARCH_K=30`
- `LME_ANSWER_TOP_K=10`
- `LME_JUDGE=longmemeval`
- `LME_JUDGE_MODEL=gpt-4o-mini`

因此你現在可直接跑：

```bash
bash scripts/run_longmemeval_service.sh full --force-reindex --no-reuse-service-ingest
```

### 什麼時候要加 `--force-reindex --no-reuse-service-ingest`

- `--force-reindex`：強制重建每題資料，避免吃到舊 ingestion 結果。
- `--no-reuse-service-ingest`：禁用既有資料重用，確保這次設定真的生效。

建議在你調整以下參數時都加上這兩個旗標：

- `LME_WRITE_MODE`
- `LME_DISTILL_BATCH`
- `LME_RESOLVER_MODE`
- `LME_DRAIN_MODE`
- embedding model 或 provider

### 主要參數說明（這次關注的組合）

| 參數 | 類型 | 預設值 | 作用 |
|---|---|---|---|
| `LME_WRITE_MODE` | 環境變數 | `distill` | ingest 寫入策略。`distill` 會走 LLM 抽取與衝突處理，通常精度高於 `raw`。 |
| `LME_DISTILL_BATCH` | 環境變數 | `2` | 每次合併幾個 session 做一次 distill。越大越快但可能犧牲召回與精度。 |
| `LME_RESOLVER_MODE` | 環境變數 | `offline` | 衝突解析模式。`offline` 代表寫入 queue，之後再處理。 |
| `LME_DRAIN_MODE` | 環境變數 | `after_run` | queue 何時清空。`after_run` 在 prepare/read 完成後統一處理。 |
| `LME_SEARCH_K` | 環境變數 | `30` | 檢索候選深度（先取多少筆候選）。通常越高召回越好，但成本增加。 |
| `LME_ANSWER_TOP_K` | 環境變數 | `10` | 實際送進回答模型的 evidence 數量。太低會漏掉正確證據。 |
| `LME_JUDGE` | 環境變數 | `longmemeval` | 評分器類型。`longmemeval` 對齊官方評分 prompt。 |
| `LME_JUDGE_MODEL` | 環境變數 | `gpt-4o-mini` | judge 使用模型；`longmemeval` 僅支援 `gpt-4o*`。 |
| `--force-reindex` | CLI 旗標 | 手動指定 | 每題重建 DB 記憶資料。適合變更配置後重新評估。 |
| `--no-reuse-service-ingest` | CLI 旗標 | 手動指定 | 不沿用既有 prepared 資料，避免結果受舊資料污染。 |

### 常用執行模式

```bash
# 1) 只做 prepare（先把資料準備好）
bash scripts/run_longmemeval_service.sh prepare --force-reindex --no-reuse-service-ingest

# 2) 只做 read + QA（重用已準備好的資料）
bash scripts/run_longmemeval_service.sh read

# 3) 全流程（prepare + read）
bash scripts/run_longmemeval_service.sh full --force-reindex --no-reuse-service-ingest
```

---

## 設定參數速查

所有參數透過 `OPENCLAW_` 前綴的環境變數設定。

### 資料庫

| 參數 | 環境變數 | 預設值 | 說明 |
|------|---------|--------|------|
| `pg_dsn` | `OPENCLAW_PG_DSN` | — | PostgreSQL 連線字串 |
| `pg_pool_min` | `OPENCLAW_PG_POOL_MIN` | 2 | 連線池最小值 |
| `pg_pool_max` | `OPENCLAW_PG_POOL_MAX` | 10 | 連線池最大值 |

### Embedding

| 參數 | 環境變數 | 預設值 | 說明 |
|------|---------|--------|------|
| `embedding_provider` | `OPENCLAW_EMBEDDING_PROVIDER` | `"auto"` | 提供者選擇 |
| `embedding_model` | `OPENCLAW_EMBEDDING_MODEL` | 依 provider | 模型名稱 |

### 搜尋

| 參數 | 環境變數 | 預設值 | 說明 |
|------|---------|--------|------|
| `search_max_results` | `OPENCLAW_SEARCH_MAX_RESULTS` | 10 | 最大結果數 |
| `search_vector_weight` | `OPENCLAW_SEARCH_VECTOR_WEIGHT` | 0.7 | 向量搜尋權重 |
| `search_text_weight` | `OPENCLAW_SEARCH_TEXT_WEIGHT` | 0.3 | 關鍵字搜尋權重 |
| `search_temporal_decay_enabled` | `OPENCLAW_SEARCH_TEMPORAL_DECAY_ENABLED` | true | 時間衰減開關 |
| `search_temporal_decay_half_life_days` | `OPENCLAW_SEARCH_TEMPORAL_DECAY_HALF_LIFE_DAYS` | 30 | 半衰期天數 |
| `search_mmr_enabled` | `OPENCLAW_SEARCH_MMR_ENABLED` | true | MMR 開關 |
| `search_mmr_lambda` | `OPENCLAW_SEARCH_MMR_LAMBDA` | 0.7 | MMR λ 值 |

### Sensory Pipeline

| 參數 | 環境變數 | 預設值 | 說明 |
|------|---------|--------|------|
| `sensory_pre_compress` | `OPENCLAW_SENSORY_PRE_COMPRESS` | true | 預壓縮開關 |
| `sensory_topic_segment` | `OPENCLAW_SENSORY_TOPIC_SEGMENT` | true | 主題分段開關 |
| `sensory_max_input_tokens` | `OPENCLAW_SENSORY_MAX_INPUT_TOKENS` | 4096 | 最大輸入 tokens |
| `sensory_topic_token_threshold` | `OPENCLAW_SENSORY_TOPIC_TOKEN_THRESHOLD` | 800 | 主題分段 token 閾值 |
| `sensory_per_message_char_limit` | `OPENCLAW_SENSORY_PER_MESSAGE_CHAR_LIMIT` | 320 | 單訊息字元上限 |

### 萃取 & 合併

| 參數 | 環境變數 | 預設值 | 說明 |
|------|---------|--------|------|
| `extraction_min_confidence` | `OPENCLAW_EXTRACTION_MIN_CONFIDENCE` | 0.5 | 最低信心值閾值 |
| `consolidation_similarity_threshold` | `OPENCLAW_CONSOLIDATION_SIMILARITY_THRESHOLD` | 0.90 | 合併相似度閾值 |
| `consolidation_max_cluster_size` | `OPENCLAW_CONSOLIDATION_MAX_CLUSTER_SIZE` | 10 | 最大叢集大小 |
| `llm_model` | `OPENCLAW_LLM_MODEL` | gpt-4o-mini | LLM 模型 |

### Working Memory

| 參數 | 環境變數 | 預設值 | 說明 |
|------|---------|--------|------|
| `working_memory_max_messages` | `OPENCLAW_WORKING_MEMORY_MAX_MESSAGES` | 20 | L1 最大訊息數 |
