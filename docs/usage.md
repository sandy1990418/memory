# openclaw-memory 使用指南

給 Chatbox 開發者的多租戶記憶系統。

---

## 目錄

1. [這個系統在做什麼](#這個系統在做什麼)
2. [三層記憶架構原理](#三層記憶架構原理)
3. [搜尋管道原理](#搜尋管道原理)
4. [寫入管道原理](#寫入管道原理)
5. [安裝與設定](#安裝與設定)
6. [實際整合到 Chatbox](#實際整合到-chatbox)
7. [完整範例：FastAPI Chatbox](#完整範例fastapi-chatbox)
8. [進階調參](#進階調參)
9. [效能數據](#效能數據)

---

## 這個系統在做什麼

你的 chatbox 每次對話都是無狀態的——LLM 不記得上一次的對話。這個系統讓你的 chatbox「記得」每個使用者說過什麼。

核心流程只有兩步：

```
使用者對話 ──寫入──▶ 從對話中萃取記憶，存進 PostgreSQL
使用者提問 ──讀取──▶ 從 PostgreSQL 搜尋相關記憶，注入 LLM prompt
```

**為什麼不直接存整段對話？** 因為對話很長、很雜。系統用 LLM 把對話「蒸餾」成結構化的記憶片段（「使用者偏好深色模式」「使用者在 2024 年 3 月換了工作」），搜尋時才能精準命中。

**為什麼需要多租戶？** 你的 chatbox 服務多個使用者，每個使用者的記憶必須完全隔離。所有查詢都帶 `user_id`，不會有跨使用者的資料洩漏。

---

## 三層記憶架構原理

人的記憶分短期和長期。這個系統仿照認知科學，把記憶分成三層：

### L1: Working Memory（工作記憶）

- **儲存位置**: Redis
- **存活時間**: 30 分鐘（TTL 自動過期）
- **內容**: 當前對話的原始訊息
- **作用**: 讓 chatbox 在同一個 session 內記得剛剛說了什麼

這是最短暫的記憶。使用者正在聊天時，最近幾句話存在 Redis 裡。session 結束就清掉。

**為什麼用 Redis？** 因為是高頻讀寫、短暫存活的資料，Redis 的 TTL 機制天生適合。而且 Redis 掛了也不影響系統——graceful degradation，直接跳過。

### L2: Episodic Memory（情節記憶）

- **儲存位置**: PostgreSQL `episodic_memories` 表
- **存活時間**: 永久，但搜尋時有 temporal decay（時間衰減）
- **內容**: 事件、session 摘要
- **作用**: 「上週三使用者抱怨了 API 速度」「上次 session 討論了部署流程」

情節記憶記的是「什麼時候發生了什麼事」。特點是越舊越不重要——半衰期 30 天，一個月前的事件在搜尋排名上只剩一半的權重。

### L3: Semantic Memory（語意記憶）

- **儲存位置**: PostgreSQL `semantic_memories` 表
- **存活時間**: 永久，沒有衰減
- **內容**: 使用者偏好、事實、決策
- **作用**: 「使用者偏好用 VS Code」「使用者的母語是中文」「使用者決定用 PostgreSQL」

語意記憶記的是「一直成立的事實」。不會因為時間而過時（除非被新的記憶覆蓋）。

### 為什麼分三層？

| 場景 | 使用的層 |
|---|---|
| 使用者說「我剛才說的那個」 | L1（工作記憶，Redis） |
| 使用者說「上次我們討論的那個部署問題」 | L2（情節記憶，temporal decay 讓最近的排前面） |
| 使用者說「你知道我喜歡什麼編輯器嗎」 | L3（語意記憶，永久保存的偏好） |

三層同時被搜尋，結果合併後返回。L1 的結果永遠排最前面（當前對話最重要）。

---

## 搜尋管道原理

當使用者提問，系統怎麼找到相關記憶？六個步驟：

### Step 1: Query Expansion（查詢擴展）

```
"使用者喜歡什麼 editor？"  →  提取關鍵字  →  ["editor"]
```

把自然語言查詢拆成關鍵字。支援中英文混合——中文用 bigram 切詞，英文去停用詞。

這一步是為了 keyword search 服務的。

### Step 2: 雙軌搜尋（Vector + Keyword）

同時跑兩種搜尋，跑四次 SQL 查詢：

**Vector Search（向量搜尋）**:
```sql
-- 把查詢文字轉成 1536 維向量，然後用 pgvector 的 <=> 運算子計算餘弦距離
SELECT content, 1.0 - (embedding <=> query_vector) AS similarity
FROM episodic_memories WHERE user_id = 'xxx'
ORDER BY embedding <=> query_vector LIMIT 20
```

向量搜尋抓的是「語意相似」——即使使用者說的詞不一樣，意思接近就會被找到。
例如：查「editor」會找到包含「VS Code」的記憶。

**Keyword Search（關鍵字搜尋）**:
```sql
-- 用 PostgreSQL 內建的 tsvector + ts_rank 做全文搜尋
SELECT content, ts_rank(tsv, plainto_tsquery('english', 'editor')) AS rank
FROM episodic_memories WHERE user_id = 'xxx'
AND tsv @@ plainto_tsquery('english', 'editor')
```

關鍵字搜尋抓的是「精確匹配」——如果使用者提到的就是那個詞，keyword 搜尋的準確率更高。

**為什麼要同時跑兩種？** 因為各有盲區：
- 向量搜尋擅長語意相似，但會找到一些「意思接近但不相關」的雜訊
- 關鍵字搜尋精確但死板，同義詞就找不到

兩種搜尋都跑在 `episodic_memories` 和 `semantic_memories` 兩張表上，所以一共四次查詢。

### Step 3: Hybrid Merge（混合合併）

```python
final_score = 0.7 * vector_score + 0.3 * keyword_score
```

把兩種搜尋的結果用加權平均合併。如果一筆記憶同時被向量和關鍵字命中，它的分數會更高。

預設權重 0.7:0.3 偏向向量搜尋，因為使用者的提問通常是自然語言而不是精確關鍵字。

### Step 4: Temporal Decay（時間衰減）

只套用在 L2 情節記憶上：

```python
decay_multiplier = exp(-λ * age_in_days)
# 半衰期 30 天：30 天前的記憶分數 × 0.5
# 60 天前的記憶分數 × 0.25
```

L3 語意記憶不衰減——「使用者偏好深色模式」不會因為是三個月前說的就不重要。

### Step 5: MMR（最大邊際相關性）

解決「搜出來的結果都差不多」的問題：

```
MMR_score = λ × relevance - (1-λ) × max_similarity_to_already_selected
```

如果前面已經選了一筆關於「VS Code」的記憶，下一筆類似的就會被壓低。讓結果更多元。

### Step 6: LLM Re-ranking（可選）

把前 20 筆候選結果丟給 LLM 重新打分。最精準但最貴——每次搜尋多一次 LLM 呼叫。預設關閉。

---

## 寫入管道原理

記憶怎麼從對話變成結構化的資料？

### 萃取（Extraction）

把整段對話丟給 LLM，用這個 prompt：

```
分析以下對話，萃取值得長期記住的事實。
每筆記憶分類為：preference / fact / decision / event
給出 confidence 分數 0-1
```

LLM 回傳結構化 JSON：

```json
[
  {"content": "使用者偏好深色模式", "memory_type": "preference", "confidence": 0.95},
  {"content": "使用者在 2024 年 3 月部署了 v2.1", "memory_type": "event", "confidence": 0.8}
]
```

### 分類路由

根據 `memory_type` 決定存到哪張表：

- `event` → `episodic_memories`（L2，有時間衰減）
- `preference` / `fact` / `decision` → `semantic_memories`（L3，永久保存）

### 嵌入 + 儲存

每筆記憶文字呼叫 embedding API 轉成 1536 維向量，然後 INSERT 到 PostgreSQL。向量存在 `embedding` 欄位，由 pgvector 的 HNSW 索引管理。

### 三個寫入時機

你的 chatbox 在三個時機應該寫入記憶：

| 時機 | 用哪個方法 | 做什麼 |
|---|---|---|
| 每次對話結束 | `ingest_conversation()` | LLM 萃取 → 分類 → 嵌入 → 存入 PG |
| Session 關閉（使用者離開） | `save_session()` | 整段對話存為一筆 episodic 記憶 + 清掉 Redis |
| Context window 快滿了 | `memory_flush()` | 緊急萃取重要記憶，防止遺忘 |

---

## 安裝與設定

### 1. 安裝 Python 套件

```bash
pip install -e ".[openai]"       # OpenAI embeddings
pip install "psycopg[binary]"    # PostgreSQL driver (psycopg3)
pip install redis                # 可選，L1 工作記憶
```

### 2. 啟動 PostgreSQL + pgvector

```bash
docker run -d --name pgvector \
  -p 5433:5432 \
  -e POSTGRES_USER=memuser \
  -e POSTGRES_PASSWORD=mempass \
  -e POSTGRES_DB=memory \
  pgvector/pgvector:pg16
```

### 3. 建立 Schema

```python
from openclaw_memory.pg_schema import get_pg_connection, ensure_pg_schema

conn = get_pg_connection("postgresql://memuser:mempass@localhost:5433/memory")
ensure_pg_schema(conn)  # 幂等，可以重複執行
conn.close()
```

這會建立三張表：

| 表 | 索引 | 用途 |
|---|---|---|
| `episodic_memories` | B-tree(user_id), HNSW(embedding), GIN(tsvector) | L2 情節記憶 |
| `semantic_memories` | 同上 | L3 語意記憶 |
| `user_profiles` | PK(user_id) | 使用者 profile（JSONB） |

### 4. 準備 LLM 函式

系統需要一個 `Callable[[str], str]` 的 LLM 函式（輸入 prompt，回傳文字）：

```python
import openai

client = openai.OpenAI()  # 自動讀 OPENAI_API_KEY

def llm_fn(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",        # 萃取用便宜的模型就夠了
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
```

---

## 實際整合到 Chatbox

以下是你的 chatbox backend 需要改動的三個地方。

### 初始化（啟動時做一次）

```python
from openclaw_memory.service import MemoryService
from openclaw_memory.embeddings import create_embedding_provider

PG_DSN = "postgresql://memuser:mempass@localhost:5433/memory"

embedding = create_embedding_provider(provider="openai")

memory_service = MemoryService(
    pg_dsn=PG_DSN,
    embedding_provider=embedding,
    llm_fn=llm_fn,                           # 上面定義的
    redis_url="redis://localhost:6379/0",     # 可選，沒有就傳 None
)
```

### 使用者發問時：搜尋記憶 → 注入 prompt

```python
def handle_user_message(user_id: str, thread_id: str, user_msg: str) -> str:
    # 1. 搜尋相關記憶
    memories = memory_service.search(
        user_id=user_id,
        query=user_msg,
        max_results=5,
        thread_id=thread_id,         # 讓 L1 也被搜尋
    )

    # 2. 把記憶塞進 system prompt
    memory_context = "\n".join(
        f"- {m.snippet}" for m in memories
    )

    messages = [
        {"role": "system", "content": f"你是一個智慧助手。\n\n使用者的相關記憶：\n{memory_context}"},
        {"role": "user", "content": user_msg},
    ]

    # 3. 呼叫 LLM
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content
```

### 對話結束時：萃取 + 儲存記憶

```python
def on_conversation_end(user_id: str, thread_id: str, conversation: list[dict]):
    # 萃取結構化記憶（LLM 呼叫）
    result = memory_service.ingest_conversation(
        user_id=user_id,
        thread_id=thread_id,
        conversation=conversation,
    )
    # result = {"inserted": 3, "skipped": 0}

    # 同時存原始 session（不經過 LLM，直接存）
    memory_service.save_session(
        user_id=user_id,
        thread_id=thread_id,
        conversation=conversation,
    )
```

就這三步。搜尋 13ms，寫入取決於 LLM 速度（通常 1-2 秒）。

---

## 完整範例：FastAPI Chatbox

```python
from fastapi import FastAPI, Request
from openclaw_memory.service import MemoryService
from openclaw_memory.embeddings import create_embedding_provider
import openai

app = FastAPI()
oai = openai.OpenAI()

def llm_fn(prompt: str) -> str:
    return oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

memory = MemoryService(
    pg_dsn="postgresql://memuser:mempass@localhost:5433/memory",
    embedding_provider=create_embedding_provider(provider="openai"),
    llm_fn=llm_fn,
)

# 每個 user 的 session 對話暫存（production 用 Redis 管理）
sessions: dict[str, list[dict]] = {}


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_id = body["user_id"]
    thread_id = body["thread_id"]
    user_msg = body["message"]

    # --- 搜尋記憶 ---
    memories = memory.search(user_id=user_id, query=user_msg, thread_id=thread_id)
    mem_text = "\n".join(f"- {m.snippet}" for m in memories) if memories else "（無相關記憶）"

    # --- 取 user profile ---
    profile = memory.get_user_profile(user_id)
    profile_text = f"使用者偏好: {profile}" if profile else ""

    # --- 組合 prompt 並呼叫 LLM ---
    system = f"你是智慧助手。\n\n{profile_text}\n\n使用者相關記憶：\n{mem_text}"
    key = f"{user_id}:{thread_id}"
    if key not in sessions:
        sessions[key] = []
    sessions[key].append({"role": "user", "content": user_msg})

    messages = [{"role": "system", "content": system}] + sessions[key]
    resp = oai.chat.completions.create(model="gpt-4o", messages=messages)
    assistant_msg = resp.choices[0].message.content

    sessions[key].append({"role": "assistant", "content": assistant_msg})
    return {"reply": assistant_msg}


@app.post("/end-session")
async def end_session(request: Request):
    body = await request.json()
    user_id = body["user_id"]
    thread_id = body["thread_id"]
    key = f"{user_id}:{thread_id}"

    conversation = sessions.pop(key, [])
    if not conversation:
        return {"status": "no conversation"}

    # --- 萃取記憶 + 存 session ---
    result = memory.ingest_conversation(user_id, thread_id, conversation)
    memory.save_session(user_id, thread_id, conversation)

    return {"status": "saved", "memories_extracted": result.get("inserted", 0)}


@app.post("/update-profile")
async def update_profile(request: Request):
    body = await request.json()
    memory.update_user_profile(body["user_id"], body["updates"])
    return {"status": "ok"}
```

### 使用方式

```bash
# 聊天
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1", "thread_id":"t1", "message":"我最近在學 Rust"}'

# 結束 session（觸發記憶萃取）
curl -X POST http://localhost:8000/end-session \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1", "thread_id":"t1"}'

# 下一次聊天——系統會記得使用者在學 Rust
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1", "thread_id":"t2", "message":"推薦我一些學習資源"}'
```

第二次聊天時，`search("推薦我一些學習資源")` 會找到「使用者在學 Rust」的記憶，注入 prompt，LLM 就能推薦 Rust 資源而不是亂猜。

---

## 進階調參

### Embedding Provider

```python
from openclaw_memory.embeddings import create_embedding_provider

# OpenAI（預設 text-embedding-3-small，1536 維）
emb = create_embedding_provider(provider="openai")

# Gemini
emb = create_embedding_provider(provider="gemini")

# Voyage AI
emb = create_embedding_provider(provider="voyage")

# 自動偵測（依環境變數）
emb = create_embedding_provider(provider="auto")
```

Schema 寫死 `vector(1536)`。如果換了不同維度的 embedding model，需要改 migration SQL。

### 搜尋參數調整

```python
from openclaw_memory.temporal_decay import TemporalDecayConfig
from openclaw_memory.mmr import MMRConfig

results = memory.search(
    user_id="u1",
    query="...",

    # 調整 vector / keyword 權重
    vector_weight=0.6,   # 降低向量影響（預設 0.7）
    text_weight=0.4,     # 提高關鍵字影響（預設 0.3）

    # 開啟時間衰減
    temporal_decay_config=TemporalDecayConfig(
        enabled=True,
        half_life_days=14.0,  # 越短 = 越偏好新記憶
    ),

    # 調整 MMR 多樣性
    mmr_config=MMRConfig(
        enabled=True,
        lambda_=0.5,  # 越低 = 結果越多樣（預設 0.7）
    ),

    # 開啟 LLM re-ranking（最精準但多一次 LLM 呼叫）
    enable_llm_rerank=True,
)
```

### 何時需要調？

| 問題 | 調什麼 |
|---|---|
| 搜出太多不相關結果 | 提高 `vector_weight`，或開啟 `enable_llm_rerank` |
| 搜出的結果都太類似 | 降低 MMR 的 `lambda_`（例如 0.4） |
| 舊的記憶排太前面 | 開啟 temporal decay 並降低 `half_life_days` |
| 使用者用精確術語搜尋 | 提高 `text_weight` |

### Batch Processing（高吞吐量場景）

如果你的 chatbox 同時服務大量使用者，可以用 batch processor 緩衝 + 批次寫入：

```python
from openclaw_memory.batch import MemoryBatchProcessor

processor = MemoryBatchProcessor(
    buffer_size=10,              # 累積 10 筆自動 flush
    llm_fn=llm_fn,
    conn=pg_connection,
    embedding_provider=embedding,
    similarity_threshold=0.85,   # 去重閾值
)

# 收到對話就丟進去
processor.buffer_conversation("user-123", messages)

# 滿 10 筆自動觸發：extract → classify → dedup → store
# 也可以手動 flush
processor.flush("user-123")

# 關機時全部 flush
processor.flush_all()
```

去重機制：新記憶和現有記憶的 cosine similarity > 0.85 時，UPDATE 而不是 INSERT。避免「使用者偏好深色模式」被存五次。

### User Profile

除了記憶搜尋，系統還提供簡單的 per-user profile CRUD：

```python
# 寫入（JSONB merge，淺合併）
memory.update_user_profile("u1", {"language": "zh-TW", "theme": "dark"})
memory.update_user_profile("u1", {"timezone": "Asia/Taipei"})

# 讀取
profile = memory.get_user_profile("u1")
# {"language": "zh-TW", "theme": "dark", "timezone": "Asia/Taipei"}
```

Profile 不走搜尋管道——直接 key-value 讀寫。適合存使用者的靜態設定。

---

## 效能數據

在 Docker pgvector + 真實 OpenAI embeddings 下的測試結果：

| 指標 | 數值 |
|---|---|
| MRR（8 queries，中英混合） | **0.938** |
| 向量搜尋延遲 | 12ms |
| 關鍵字搜尋延遲 | 1.6ms |
| 完整搜尋管道平均延遲 | **13ms** |
| 完整搜尋管道 P95 | 15ms |
| Profile 讀取延遲 | 8ms |
| 使用者隔離 | 已驗證（0 筆跨使用者洩漏） |
| 中文搜尋 | 已驗證（rank-1 命中） |
| 規模測試（100 筆記憶, 10 使用者） | 11ms avg |

### 怎麼跑 integration test

```bash
# 起 pgvector
docker compose -f docker-compose.test.yml up -d

# 等 health check
docker compose -f docker-compose.test.yml ps

# 建 schema
psql "postgresql://testuser:testpass@localhost:5433/memory_test" \
  -f migrations/001_initial_schema.sql

# 跑測試
OPENAI_API_KEY="sk-..." python -m pytest tests/integration/test_real_pg.py -v

# 清除
docker compose -f docker-compose.test.yml down -v
```

---

## 模組對照表

| 模組 | 用途 | 你需要直接用嗎？ |
|---|---|---|
| `service.py` | 統一入口（MemoryService） | **是，這是你唯一需要的 API** |
| `embeddings.py` | Embedding provider 建立 | 是，初始化時用一次 |
| `pg_schema.py` | 建表、連線 | 是，初始化時用一次 |
| `extraction.py` | LLM 記憶萃取 | 不用，service 內部呼叫 |
| `pg_search.py` | PostgreSQL 向量/關鍵字搜尋 | 不用，service 內部呼叫 |
| `hybrid.py` | 混合合併 | 不用，service 內部呼叫 |
| `temporal_decay.py` | 時間衰減 | 不用，但 config 類別會用到 |
| `mmr.py` | MMR 多樣性排序 | 不用，但 config 類別會用到 |
| `llm_rerank.py` | LLM 重排序 | 不用，service 內部呼叫 |
| `query_expansion.py` | 查詢擴展 | 不用，service 內部呼叫 |
| `working_memory.py` | Redis L1 | 不用，service 內部管理 |
| `dedup.py` | 去重 | 不用，batch processor 內部呼叫 |
| `batch.py` | 批次處理 | 高吞吐量才需要，見進階調參 |
