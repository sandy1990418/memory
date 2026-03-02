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
# 推薦：用專案內建 compose + helper script
bash scripts/pgvector_local.sh start
bash scripts/pgvector_local.sh init
export OPENCLAW_PG_DSN="postgresql://memuser:mempass@localhost:5433/memory"
```

常用管理指令：

- `bash scripts/pgvector_local.sh status`
- `bash scripts/pgvector_local.sh logs`
- `bash scripts/pgvector_local.sh stop`（保留資料）
- `bash scripts/pgvector_local.sh reset`（刪除資料）

### 3. 建立 Schema

```python
from openclaw_memory.pg_schema import get_pg_connection, ensure_pg_schema

conn = get_pg_connection("postgresql://memuser:mempass@localhost:5433/memory")
ensure_pg_schema(conn)  # 幂等，可以重複執行
conn.close()
```

這會建立主要記憶表（含 resolver queue）：

| 表 | 索引 | 用途 |
|---|---|---|
| `episodic_memories` | B-tree(user_id), HNSW(embedding), GIN(tsvector) | L2 情節記憶 |
| `semantic_memories` | 同上 | L3 語意記憶 |
| `canonical_memories` | active key unique, HNSW(embedding) | resolver 後的 canonical 記憶 |
| `memory_update_queue` | (status, available_at), (user_id, status) | offline resolver 更新佇列 |
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

### Three-Layer Memory Pipeline (Structured Distillation, Conflict Resolution, Answer Contract)

openclaw-memory supports an optional three-layer write/read pipeline that adds structured intelligence on top of the basic extract-and-store flow. Each layer is independently controlled by a config flag and degrades gracefully if the underlying LLM or parser fails.

#### Pipeline Overview

```
Ingest conversation
       │
       ▼
  [Layer 1] Structured Distillation (enable_structured_distill)
       │   LLM extracts typed memory facts with confidence scores.
       │   Falls back to raw text extraction on parse failure.
       │
       ▼
  [Layer 2] Conflict Resolver (enable_conflict_resolver)
       │   Detects contradictions between new and existing memories.
       │   Resolves conflicts by updating or superseding stale records.
       │   Falls back to simple upsert on LLM/parser failure.
       │
       ▼
  [Layer 3] Answer Contract (enable_answer_contract)
           Wraps retrieval results with an evidence contract —
           each claim in the answer is tied back to a source chunk.
           Falls back to plain retrieval results on failure.
```

#### Config Flags

| Flag | Type | Default | Effect |
|---|---|---|---|
| `enable_structured_distill` | `bool` | `False` | Enable Layer 1: LLM-based structured extraction of typed memory facts |
| `enable_conflict_resolver` | `bool` | `False` | Enable Layer 2: detect and resolve contradictions with existing memories |
| `enable_answer_contract` | `bool` | `False` | Enable Layer 3: evidence-backed answer contract on retrieval results |

Pass these flags via `MemoryService` constructor or the config overrides dict:

```python
from openclaw_memory.service import MemoryService

memory_service = MemoryService(
    pg_dsn="postgresql://memuser:mempass@localhost:5433/memory",
    embedding_provider=embedding,
    llm_fn=llm_fn,
    enable_structured_distill=True,
    enable_conflict_resolver=True,
    enable_answer_contract=True,
)
```

Or via `resolve_memory_search_config` overrides in benchmark scripts:

```python
overrides = {
    "enable_structured_distill": True,
    "enable_conflict_resolver": True,
    "enable_answer_contract": True,
}
```

#### Layer 1: Structured Distillation

When `enable_structured_distill=True`, the extraction step calls an LLM with a structured prompt that returns typed memory facts:

```json
[
  {"content": "User prefers dark mode", "memory_type": "preference", "confidence": 0.95},
  {"content": "User deployed v2.1 in March 2024", "memory_type": "event", "confidence": 0.8}
]
```

Each fact is assigned a `memory_type` that controls routing to `episodic_memories` (events) or `semantic_memories` (preferences/facts/decisions). Confidence scores below a threshold (default 0.5) are dropped.

**Fallback behavior**: If the LLM returns unparseable JSON or times out, the system falls back to the basic extraction path (raw text chunking), ensuring memories are always stored even when the structured distillation fails.

#### Layer 2: Conflict Resolver

When `enable_conflict_resolver=True`, before inserting new memories the system:

1. Searches existing memories for semantically similar content (cosine similarity > 0.85)
2. Passes both old and new memory to an LLM conflict-detection prompt
3. If a contradiction is detected, marks the old memory as superseded and stores the new one

Example: if the user previously said "I use VSCode" and now says "I switched to Neovim", the resolver will update the stored preference rather than creating a duplicate.

**Fallback behavior**: If conflict resolution fails (LLM error, parser error, or database timeout), the system falls back to the deduplication logic: update if similarity > threshold, insert otherwise.

#### Layer 3: Answer Contract

When `enable_answer_contract=True`, retrieval results are augmented with an evidence contract — each retrieved chunk is annotated with its support for the final answer:

```python
memories = memory_service.search(
    user_id="u1",
    query="What editor does the user prefer?",
    enable_answer_contract=True,
)
# Each result in memories now includes:
# result.evidence_supported  -> True/False (heuristic: retrieval hit + non-empty)
# result.evidence_source     -> source chunk path
```

This layer is used by the benchmark to compute `evidence_supported_rate`, `unsupported_claim_rate`, and `abstention_precision`.

**Fallback behavior**: If the answer contract computation fails, the system returns plain retrieval results without evidence annotations. The `evidence_*` fields default to `None` in the output JSON.

#### Enabling Layers Selectively

You can enable any combination of layers. Typical configurations:

| Use case | Recommended flags |
|---|---|
| Production (balanced cost/quality) | `enable_structured_distill=True` |
| High-accuracy recall + dedup | `enable_structured_distill=True, enable_conflict_resolver=True` |
| Full pipeline with evidence | All three enabled |
| Benchmark evaluation only | `enable_answer_contract=True` (no write-time overhead) |

#### LightMem-Style Speed/Token Optimizations

`MemoryService` now supports a LightMem-inspired write-path optimization profile:

```python
memory_service = MemoryService(
    pg_dsn=PG_DSN,
    embedding_provider=embedding,
    llm_fn=llm_fn,
    enable_structured_distill=True,
    enable_conflict_resolver=True,
    enable_lightmem=True,            # turn on the profile
    resolver_update_mode="offline",  # sleep-time canonical consolidation
    save_session_mode="summary",     # store compact session text instead of full raw log
)
```

New constructor knobs:

| Flag | Type | Default | Effect |
|---|---|---|---|
| `enable_lightmem` | `bool` | `False` | Enables recommended defaults for fast/low-token write path |
| `pre_compress` | `bool \| None` | `None` | Pre-compresses long turns before extraction prompt build |
| `messages_use` | `str \| None` | `None` | Role filter (`\"all\"` or `\"user_only\"`) before extraction |
| `topic_segment` | `bool \| None` | `None` | Topic-based segmentation for long conversations |
| `max_distill_tokens` | `int` | `2200` | Hard token budget for extraction input |
| `topic_token_threshold` | `int` | `600` | Segment split threshold (estimated tokens) |
| `distill_min_confidence` | `float` | `0.0` | Drop extracted memories below this confidence |
| `resolver_update_mode` | `str` | `\"sync\"` | `\"sync\"` or `\"offline\"` conflict resolver execution |
| `save_session_mode` | `str` | `\"raw\"` | `\"raw\"` or `\"summary\"` session archival strategy |
| `session_summary_chars` | `int` | `1800` | Character cap when `save_session_mode=\"summary\"` |

When `resolver_update_mode=\"offline\"`, writes enqueue resolver jobs into
`memory_update_queue` and return quickly. Run a background cycle:

```python
stats = memory_service.drain_update_queue(limit=200)
print(stats)  # {"claimed": ..., "processed": ..., "retried": ..., "failed": ...}
```

### Batch Processing（高吞吐量場景）

如果你的 chatbox 同時服務大量使用者，可以用 batch processor 緩衝 + 批次寫入：

```python
from openclaw_memory.batch import MemoryBatchProcessor

processor = MemoryBatchProcessor(
    buffer_size=10,              # 累積 10 筆自動 flush
    token_buffer_threshold=1200, # 或累積 token 達門檻時自動 flush
    llm_fn=llm_fn,
    conn=pg_connection,
    embedding_provider=embedding,
    similarity_threshold=0.85,   # 去重閾值
    distill_pre_compress=True,   # LightMem 風格前置壓縮
    distill_messages_use="user_only",
    distill_topic_segment=True,
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


---

## DB 工作記憶（無 Redis 環境）

當你的環境**沒有 Redis**，或你不想管理 Redis，系統會自動使用 PostgreSQL 作為 L1 工作記憶。這個功能叫做 **DB Working Memory**（`DBWorkingMemory`）。

### 特性說明

| 特性 | 說明 |
|---|---|
| 儲存位置 | PostgreSQL `working_messages` 表 |
| 作用域 | 僅 `user_id`（**無** `thread_id` 分隔） |
| 預設上限 | **N=20** 條最近訊息（可透過 `working_memory_limit` 覆寫） |
| 自動啟用 | 只要有 `pg_dsn`，不需要額外設定 |
| 排序 | 舊到新（oldest-first），方便直接塞入 LLM prompt |

> **注意**：DB 工作記憶以 `user_id` 為唯一維度，**不支援** `thread_id` 隔離。若需要 thread 層級的隔離，請使用 Redis 工作記憶。

### 自動建立

只要你在初始化 `MemoryService` 時傳入 `pg_dsn`，系統就會**自動建立** `DBWorkingMemory`，不需要傳 `redis_url`：

```python
from openclaw_memory.service import MemoryService
from openclaw_memory.embeddings import create_embedding_provider

memory_service = MemoryService(
    pg_dsn="postgresql://memuser:mempass@localhost:5433/memory",
    embedding_provider=create_embedding_provider(provider="openai"),
    llm_fn=llm_fn,
    # redis_url 不傳 → 自動使用 DB 工作記憶
)
```

### 寫入：record_message()

每當使用者或助手說了一句話，呼叫 `record_message()` 把它存進 DB 工作記憶：

```python
# 使用者說話
memory_service.record_message(
    user_id="user-123",
    role="user",
    content="我想用 Python 寫一個爬蟲",
)

# 助手回覆
memory_service.record_message(
    user_id="user-123",
    role="assistant",
    content="好的，我可以幫你寫一個用 requests + BeautifulSoup 的爬蟲",
)
```

- `role` 通常是 `"user"` 或 `"assistant"`，也可以是任意字串
- 每次 append 後，系統自動保留最近 20 條，舊的自動刪除
- 若 DB 工作記憶未初始化，此方法**靜默 no-op**（不會拋例外）

### 搜尋時自動注入工作記憶

呼叫 `search()` 時，系統自動把最近 N=20 條訊息**前置**在搜尋結果最前面（score 固定為 1.0，永遠排最前）：

```python
memories = memory_service.search(
    user_id="user-123",
    query="爬蟲要用什麼函式庫？",
    max_results=5,
    # include_working_memory=True  (預設開啟)
    # working_memory_limit=20      (預設 20，可覆寫)
)
```

限制取用的工作記憶條數：

```python
memories = memory_service.search(
    user_id="user-123",
    query="...",
    working_memory_limit=5,   # 只取最近 5 條
)
```

完全跳過工作記憶：

```python
memories = memory_service.search(
    user_id="user-123",
    query="...",
    include_working_memory=False,
)
```

### Redis 與 DB 的優先順序

| 條件 | 使用的工作記憶 |
|---|---|
| 傳了 `redis_url` + 傳了 `thread_id` | Redis（thread 隔離） |
| 沒有 Redis，有 `pg_dsn` | DB 工作記憶（user_id 隔離，N=20） |
| 兩者都沒有 | 無工作記憶（僅 L2/L3） |

### 整合範例

```python
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_id = body["user_id"]
    user_msg = body["message"]

    # 1. 把使用者訊息寫入 DB 工作記憶
    memory_service.record_message(user_id, "user", user_msg)

    # 2. 搜尋記憶（DB 工作記憶自動前置，最多 20 條）
    memories = memory_service.search(user_id=user_id, query=user_msg)
    mem_text = "\n".join(f"- {m.snippet}" for m in memories)

    # 3. 呼叫 LLM
    resp = oai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"相關記憶：\n{mem_text}"},
            {"role": "user", "content": user_msg},
        ],
    )
    assistant_msg = resp.choices[0].message.content

    # 4. 把助手回覆也存入工作記憶
    memory_service.record_message(user_id, "assistant", assistant_msg)

    return {"reply": assistant_msg}
```

---

## Benchmark Sweep（評測與調參）

`tests/benchmark/run_sweep.py` 是內建的 benchmark 執行器，用來比較不同參數組合對搜尋品質（MRR、Recall@5、nDCG@5）的影響。指標定義參考 `docs/benchmark-research.md` §3.1。

### 快速執行

```bash
# 執行所有 sweep（weights + mmr + decay）
python tests/benchmark/run_sweep.py

# 只執行特定 sweep 維度
python tests/benchmark/run_sweep.py --sweep weights
python tests/benchmark/run_sweep.py --sweep mmr
python tests/benchmark/run_sweep.py --sweep decay

# 同時執行多個維度
python tests/benchmark/run_sweep.py --sweep weights mmr
```

### Sweep 類型

| Sweep 類型 | 調整的參數 | 掃描範圍 |
|---|---|---|
| `weights` | `vector_weight` / `text_weight` | vw ∈ {0.3, 0.4, ..., 0.9}，tw = 1 - vw |
| `mmr` | `mmr_lambda` | λ ∈ {0.3, 0.5, 0.7, 0.9, 1.0} |
| `decay` | `half_life_days` | disabled, 7, 14, 30, 60, 90 天 |
| `all` | 三者全跑 | — |

### 結果輸出

Sweep 完成後輸出摘要表格（stdout）和可選 JSON 報告：

```bash
# 輸出到 JSON 檔案
python tests/benchmark/run_sweep.py --output sweep_results.json
```

JSON 格式範例：

```json
{
  "sweep": ["all"],
  "mode": "golden",
  "queries_count": 12,
  "results": [
    {
      "label": "hybrid vw=0.7 tw=0.3",
      "sweep_type": "weights",
      "params": {"vector_weight": 0.7, "text_weight": 0.3},
      "aggregate": {"mrr": 0.875, "recall@5": 0.875, "ndcg@5": 0.891},
      "elapsed_s": 1.2
    }
  ]
}
```

### 執行個別 benchmark runner

```bash
# Golden corpus benchmark（mock embeddings，不需要 API key）
python tests/benchmark/run_sweep.py --run-golden

# LongMemEval benchmark（最多 50 筆）
python tests/benchmark/run_sweep.py --run-benchmark --benchmark-limit 50

# 真實 embedding benchmark（需要 OPENAI_API_KEY）
python tests/benchmark/run_sweep.py --run-real-embedding

# LoCoMo benchmark（需要先下載 locomo10.json）
python tests/benchmark/run_locomo_benchmark.py --limit 60

# LongMemEval QA（端到端：檢索 + 回答，需 OPENAI_API_KEY）
# 預設使用官方 LongMemEval judge（gpt-4o-mini），回答模型預設 gpt-5-mini
python tests/benchmark/run_longmemeval_qa.py --limit 48 --balanced

# 使用快取（預設會把每個 instance 的 DB 存在 tests/benchmark/.cache/）
# 若要關閉快取或強制重建：
python tests/benchmark/run_longmemeval_qa.py --limit 48 --balanced --no-cache
python tests/benchmark/run_longmemeval_qa.py --limit 48 --balanced --force-reindex

# LongMemEval QA baseline 對照（fts + mock + openai）
python tests/benchmark/run_longmemeval_qa.py --limit 48 --balanced --provider all

# 產生 QA 後直接更新報告
python tests/benchmark/run_longmemeval_qa.py --limit 48 --balanced --update-report

# 使用官方 LongMemEval judge（對齊官方評測）
python tests/benchmark/run_longmemeval_qa.py --limit 48 --balanced --judge longmemeval --judge-model gpt-4o-mini

# 建議：拉高 QA 證據覆蓋（多 session / temporal 題型）
python tests/benchmark/run_longmemeval_qa.py \
  --limit 48 --balanced \
  --search-k 30 \
  --answer-top-k 10 \
  --context-lines 60 \
  --diversify-paths

# 使用 MemoryService pipeline（PostgreSQL；測 service read path）
python tests/benchmark/run_longmemeval_qa.py \
  --pipeline service \
  --pg-dsn "$OPENCLAW_PG_DSN" \
  --provider openai \
  --config hybrid \
  --limit 48 --balanced

# 使用完整三層 write path（distill + resolver + answer）
python tests/benchmark/run_longmemeval_qa.py \
  --pipeline service \
  --service-write-mode distill \
  --distill-batch-sessions 8 \
  --service-lightmem \
  --service-resolver-mode offline \
  --service-drain-queue-mode after_run \
  --reuse-service-ingest \
  --pg-dsn "$OPENCLAW_PG_DSN" \
  --provider openai \
  --config hybrid \
  --limit 48 --balanced

# 兩階段模式（推薦）：
# 1) 先 prepare 一次（只做寫入，不跑 QA，最快）
python tests/benchmark/run_longmemeval_qa.py \
  --pipeline service \
  --prepare-only \
  --service-write-mode distill \
  --distill-batch-sessions 8 \
  --service-workers 4 \
  --answer-model gpt-4o-mini \
  --service-resolver-mode off \
  --service-drain-queue-mode never \
  --reuse-service-ingest \
  --pg-dsn "$OPENCLAW_PG_DSN" \
  --provider openai \
  --limit 48 --balanced

# 2) 後續只跑 read + answer（跳過 ingest）
# 注意：distill-batch-sessions 必須跟 prepare 相同
python tests/benchmark/run_longmemeval_qa.py \
  --pipeline service \
  --read-answer-only \
  --service-write-mode distill \
  --distill-batch-sessions 8 \
  --service-workers 4 \
  --answer-model gpt-5-mini \
  --service-resolver-mode off \
  --service-drain-queue-mode never \
  --pg-dsn "$OPENCLAW_PG_DSN" \
  --provider openai \
  --config hybrid \
  --limit 48 --balanced

# 一鍵跑兩階段（推薦，避免參數不一致）
bash scripts/run_longmemeval_service.sh full
```

`scripts/run_longmemeval_service.sh` 可用環境變數覆蓋預設：

- `OPENCLAW_PG_DSN`（預設 `postgresql://memuser:mempass@localhost:5433/memory`）
- `LME_LIMIT`（預設 `48`）
- `LME_DISTILL_BATCH`（預設 `8`）
- `LME_PREP_MODEL`（預設 `gpt-4o-mini`）
- `LME_QA_MODEL`（預設 `gpt-5-mini`）
- `LME_RESOLVER_MODE`（預設 `off`）
- `LME_DRAIN_MODE`（預設 `never`）
- `LME_SERVICE_WORKERS`（預設 `2`，平行 prepare worker 數）

若覺得 `service + distill` 太慢，可先用：

- `--distill-batch-sessions 8`（降低 extraction 呼叫數）
- `--service-workers 2~4`（平行跑 prepare，各題不用等上一題）
- `--answer-model gpt-4o-mini`（prepare 階段更快）
- `--service-resolver-mode off`（先關 resolver，等需要再開）
- `--reuse-service-ingest`（重跑時重用已寫入資料，避免每次重做 distill）
- `--judge exact`（先關掉 LLM judge）
- `--limit 6`（小樣本快速迭代）

LongMemEval QA 指標解讀（重要）：

- `retrieval_hit@5`: top-5 命中任一證據檔（寬鬆）
- `retrieval_coverage@5`: top-5 覆蓋到的證據檔比例（較真實）
- `retrieval_all_hit@5`: top-5 是否覆蓋所有證據檔（最嚴格）

若 `retrieval_hit@5` 高、但 `retrieval_all_hit@5` 低，代表多半是「證據不完整」而非純回答模型問題。

LoCoMo 下載（官方資料集）：

```bash
curl -L -o tests/benchmark/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

### 使用真實 Embedding（OpenAI API Key）

真實 embedding benchmark 需要 `OPENAI_API_KEY`。系統支援從 `.env` 自動載入，**不需要額外安裝套件**：

```bash
# 在 repo 根目錄建立 .env（請勿 commit）
echo "OPENAI_API_KEY=sk-..." > .env

# 執行（系統自動讀取 .env）
python tests/benchmark/run_sweep.py --run-real-embedding

# 只跑 OpenAI（跳過 mock/fts）
python tests/benchmark/run_real_embedding_benchmark.py --only-openai --longmemeval --limit 50
```

也可以明確指定 `.env` 路徑：

```bash
python tests/benchmark/run_sweep.py --run-real-embedding --dotenv /path/to/.env
```

若環境變數已存在則 `.env` 不會覆蓋它。`.env` 已在 `.gitignore` 中排除，不會被 commit。

---

## Quality Commands（品質指令）

在 commit 或 PR 前，建議執行以下指令確保程式碼品質。

### Lint（ruff）

```bash
ruff check src/ tests/
```

自動修復可修正的問題：

```bash
ruff check --fix src/ tests/
```

### Type Check（mypy）

```bash
mypy src/openclaw_memory/
```

### Unit Tests（pytest）

```bash
# 所有單元測試（不需要 DB，快速）
pytest tests/unit/ -v

# 特定測試檔
pytest tests/unit/test_working_memory.py -v
```

### Integration Tests（需要 PostgreSQL）

```bash
# 啟動測試用 pgvector（ephemeral）
docker compose -f docker-compose.test.yml up -d

# 執行 integration tests
pytest tests/integration/ -v

# 清除
docker compose -f docker-compose.test.yml down -v
```

### Service 三層流程測試（LongMemEval QA）

```bash
# 1) 啟動本地 PG（persistent）
bash scripts/pgvector_local.sh start
bash scripts/pgvector_local.sh init
export OPENCLAW_PG_DSN="postgresql://memuser:mempass@localhost:5433/memory"

# 2) 小樣本 smoke test（先寫入）
LME_LIMIT=6 LME_DISTILL_BATCH=8 bash scripts/run_longmemeval_service.sh prepare

# 3) 小樣本讀取 + 回答（不重做 ingest）
LME_LIMIT=6 LME_DISTILL_BATCH=8 bash scripts/run_longmemeval_service.sh read
```

### 一次執行所有品質檢查

```bash
ruff check src/ tests/ && mypy src/openclaw_memory/ && pytest tests/unit/ -v
```

若 repo 內有 `scripts/quality.sh`，也可以直接執行：

```bash
bash scripts/quality.sh
```

`scripts/quality.sh` 會依序執行：ruff → mypy → pytest → benchmark。
**所有步驟都必須通過**，quality gate 才算成功。
注意：benchmark 目前使用 LongMemEval + OpenAI embeddings，必須提供
`OPENAI_API_KEY`（可從 `.env` 載入）。

---

## Enforced Benchmarks（品質門檻中的 Benchmark）

Benchmark 是品質工作流程的一部分，由 `scripts/quality.sh` 自動觸發。

### scripts/benchmark.sh

```bash
# 手動執行 benchmark（需要 OPENAI_API_KEY）
bash scripts/benchmark.sh
```

這個腳本會：
1. 執行 LongMemEval（OpenAI embeddings）：
   `python tests/benchmark/run_real_embedding_benchmark.py --longmemeval --limit 50`
2. 將結果寫入 `tests/benchmark/results_real_vs_mock.json`
3. 呼叫報告產生器，更新 `docs/benchmark-report.md`

**這是「真實 benchmark」流程**，需要 `OPENAI_API_KEY`（可從 `.env` 載入）。

### docs/benchmark-report.md（自動產生）

`docs/benchmark-report.md` 是由 `scripts/generate_benchmark_report.py` 自動產生的，**不要手動編輯**。
每次執行 `scripts/benchmark.sh` 都會重新產生，內容包含：

- 產生時間戳
- LongMemEval 的 OpenAI 結果摘要
- 全部 LongMemEval 結果表（含 provider / config）
- OpenAI 的 per-type breakdown
- 若存在 `tests/benchmark/results_longmemeval_qa.json`，會額外包含 LongMemEval QA（端到端）結果

手動觸發報告產生：

```bash
python scripts/generate_benchmark_report.py \
    --input tests/benchmark/results_real_vs_mock.json \
    --output docs/benchmark-report.md \
    --qa-input tests/benchmark/results_longmemeval_qa.json
```

### OpenAI API Key（必要）

請在 repo 根目錄建立 `.env`（請勿 commit）：

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

執行 benchmark：

```bash
bash scripts/benchmark.sh
```
