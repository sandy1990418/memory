# OpenClaw Memory — Code Wiki

> 基於原始碼完整分析產生，非抄寫既有文件

---

## 目錄

1. [系統總覽](#系統總覽)
2. [目錄結構](#目錄結構)
3. [架構圖](#架構圖)
4. [記憶生命週期引擎](#記憶生命週期引擎)
5. [Token 監控與預算](#token-監控與預算)
6. [三層記憶模型](#三層記憶模型)
7. [兩條主要管線](#兩條主要管線)
8. [模組詳解](#模組詳解)
9. [資料庫 Schema](#資料庫-schema)
10. [API 端點一覽](#api-端點一覽)
11. [使用指南：如何跑起來](#使用指南如何跑起來)
12. [使用指南：常見操作](#使用指南常見操作)
13. [使用指南：LongMemEval Benchmark Service](#使用指南longmemeval-benchmark-service)
14. [設定參數速查](#設定參數速查)

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

### 資料流全景圖

```
═══════════════════════════════════════════════════════════════════════════════
                            寫 入 路 徑（Ingest）
═══════════════════════════════════════════════════════════════════════════════

  session/start ──→ 初始化 L1 + 清理孤兒 session（> 2h 未活動）

  session/message ──→ append to L1 (working_messages)
                          │
                          ├─ 每 8 則（buffer_drain_every）自動觸發 ↓
                          │
                     ┌────▼─────────────────────────────────────────┐
                     │         Mid-Session Extraction               │
                     │                                              │
                     │  Sensory Filter ──→ LLM Extraction ──→ Conflict Resolution
                     │  (過濾/壓縮/分段)    (萃取結構化記憶)    (比對 canonical)   │
                     │                                         │              │
                     │                           ┌─────────────┼──────┐       │
                     │                           ▼             ▼      ▼       │
                     │                    event → L2    fact/pref → L3  canonical
                     │                    (episodic)    (semantic)   (ADD/UPDATE/
                     │                                               SUPERSEDE/
                     │                                               DELETE/NOOP)
                     │                                                        │
                     │  ※ 不清除 L1，session 繼續                             │
                     └────────────────────────────────────────────────────────┘

  session/end ──→ 取殘餘訊息（上次 drain 之後）──→ 同上 extraction 流程
                     │
                     ├─ 全 session 摘要 ──→ L2 episodic (session episode)
                     │
                     ├─ 清除 L1 + drain offset
                     │
                     ├─ Consolidation 檢查（新 canonical >= 20 筆？）
                     │       │
                     │       └─ YES → Background Thread（advisory lock）
                     │                  ├─ Cluster（cosine >= 0.90）
                     │                  ├─ LLM Merge → 標記舊為 superseded
                     │                  ├─ Staleness Detection → delete/abstract
                     │                  └─ Promotion（event > 90天 → fact）
                     │
                     └─ Lazy Cleanup
                            ├─ 刪除孤兒 working messages
                            └─ 物理刪除 superseded/deleted > 30天

  memory/add ──→ 直接走 Extraction pipeline（或直接插入單筆記憶）

═══════════════════════════════════════════════════════════════════════════════
                            讀 取 路 徑（Retrieval）
═══════════════════════════════════════════════════════════════════════════════

  ★ search/answer（chatbot 主路徑）
      │
      ├─ 預算檢查（SELECT ... FOR UPDATE 原子鎖）
      │     超額 → abstain: "Daily token budget exceeded"
      │
      └──→ service.search(include_working=有session_id時為True)
              │
              │  ┌─────────── ① pipeline_search() ─────────────────────┐
              │  │                                                      │
              │  │  for tbl in [episodic, semantic, canonical]:         │
              │  │    ├─ search_vector(query_vec)  ← cosine, HNSW      │
              │  │    └─ search_keyword(query)     ← BM25, GIN*        │
              │  │                                                      │
              │  │  → hybrid merge (0.7 × vector + 0.3 × text)        │
              │  │  → temporal decay (半衰期 30天)                      │
              │  │  → MMR 多樣性重排 (λ=0.7)                           │
              │  │  → LLM Rerank（可選）                                │
              │  │  → 排序後的 L2/L3 結果                               │
              │  └──────────────────────────────────────────────────────┘
              │
              │  ┌─────────── ② L1 Working Memory ─────────────────────┐
              │  │  （僅在有 session_id 時）                             │
              │  │  get_recent() → 該 session 全部 raw messages        │
              │  │  → 不做搜尋，全部塞入，score 固定 1.0（最高優先）   │
              │  └──────────────────────────────────────────────────────┘
              │
              └──→ [L1 結果] + [L2/L3 結果] → 截斷到 top_k（預設 6）
                          │
                          ▼
                  ┌─ generate_answer() ─────────────────────────────┐
                  │                                                  │
                  │  LLM 收到的 prompt:                              │
                  │  ┌────────────────────────────────────────────┐  │
                  │  │ ## User question                           │  │
                  │  │ 使用者喜歡吃什麼？                         │  │
                  │  │                                            │  │
                  │  │ ## Evidence                                │  │
                  │  │ [1] id='wm:...'  score=1.000  ← L1 raw    │  │
                  │  │ 我今天中午吃了拉麵很好吃                   │  │
                  │  │                                            │  │
                  │  │ [2] id='uuid'  score=0.823  ← L2 episodic │  │
                  │  │ 使用者 2024-12 提到喜歡吃拉麵              │  │
                  │  │                                            │  │
                  │  │ [3] id='uuid'  score=0.791  ← L3 canonical│  │
                  │  │ 使用者偏好拉麵                              │  │
                  │  │                                            │  │
                  │  │ ... 截斷到 top_k 筆                        │  │
                  │  └────────────────────────────────────────────┘  │
                  │                                                  │
                  │  ※ LLM 不知道哪筆來自哪層                       │
                  │    只看到 numbered evidence + score              │
                  │                                                  │
                  │  → LLM 回傳 JSON:                                │
                  │    {answer, evidence[{memory_id, quote, reason}], │
                  │     confidence, abstain}                          │
                  │                                                  │
                  │  → 若 LLM abstain 但有 evidence → 自動 retry     │
                  └──────────────────────────────────────────────────┘

          * canonical_memories 無 GIN 索引，keyword 即時計算 to_tsvector
          ⚠ L1 的 score=1.0 且放最前面，若 L1 訊息多（>top_k），
            L2/L3 長期記憶會被擠掉。僅在傳 session_id 時發生。

═══════════════════════════════════════════════════════════════════════════════
                            儲 存 層
═══════════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │                        PostgreSQL + pgvector                        │
  │                                                                      │
  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
  │  │ working_messages │  │episodic_memories│  │ semantic_memories   │  │
  │  │    （L1）        │  │    （L2）        │  │    （L3）           │  │
  │  │ session 範圍     │  │ 時間衰減        │  │ 永久儲存           │  │
  │  │ 最多 20 筆      │  │ HNSW + GIN      │  │ HNSW + GIN         │  │
  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
  │                                                                      │
  │  ┌─────────────────────────────┐  ┌──────────────────────────────┐  │
  │  │     canonical_memories      │  │       user_profiles          │  │
  │  │ 合併後事實（unique key/user）│  │ token_budget_daily: 100000   │  │
  │  │ status: active/superseded/  │  │ tokens_used_today（每日重置） │  │
  │  │         deleted             │  │ budget_reset_at              │  │
  │  │ HNSW index                  │  └──────────────────────────────┘  │
  │  └─────────────────────────────┘                                    │
  │                                                                      │
  │  vector 維度由 embedding_dimensions 設定（預設 1536）                │
  └──────────────────────────────────────────────────────────────────────┘
```

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

### 搜尋端點使用場景

系統提供兩種搜尋路徑，服務不同場景：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ★ Chatbot 主路徑（核心場景）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 POST /search/answer
      │
      └─→ service.answer()
              │
              └─→ service.search()  ← 完整 hybrid search
                      │
                      ├─ search_vector() × 3 表（cosine）
                      ├─ search_keyword() × 3 表（BM25）
                      ├─ hybrid merge（0.7/0.3）
                      ├─ temporal decay + MMR
                      └─ Top-K full content
                              │
                              └─→ generate_answer()  ← RAG，直接用完整內容

 ※ 不經過分層搜尋，一步到位拿完整內容 → LLM 生成答案

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 預留端點（Admin Dashboard / MCP Agent / 外部整合用）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 POST /search          → Tier 1: id + title + score（~50 tokens/筆）
 GET  /search/timeline → Tier 2: content + 時序鄰居（~200 tokens/筆）
 POST /search/detail   → Tier 3: full content + metadata

 這三個端點實作了 Progressive Disclosure 模式（類似 claude-mem），
 適合需要逐步探索的場景（如 AI agent 工具呼叫、admin 介面瀏覽）。
 目前 chatbot 主路徑不使用這些端點。
```

### Session 生命週期（含 Buffer Drain）

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    participant L1 as L1 Working
    participant L2 as L2 Episodic
    participant L3 as L3 Semantic
    participant BG as Background Thread

    C->>S: POST /session/start {user_id, session_id}
    S->>L1: 初始化 session
    S->>L1: 清理該 user 的孤兒 session（> 2h 未活動）

    loop 對話進行中
        C->>S: POST /session/message {role, content}
        S->>L1: append(role, content)
        alt 每 8 則訊息（buffer_drain_every）
            S->>S: Sensory → Extract → Conflict Resolve（僅新訊息）
            S->>L2: store_episodic（事件類）
            S->>L3: store_semantic（事實/偏好/決策）
            Note right of S: 不清除 L1，session 繼續
        end
    end

    C->>S: POST /session/end {user_id, session_id}
    S->>L1: 讀取上次 drain 之後的殘餘訊息
    S->>S: Sensory → Extract → Conflict Resolve（殘餘）
    S->>L2: store_episodic + session summary
    S->>L3: store_semantic
    S->>L1: clear_session

    alt 累積 >= 20 筆新 canonical 記憶
        S->>BG: 啟動 background consolidation
        BG->>BG: cluster → merge → promote → dedup
    end

    S->>S: lazy cleanup（superseded 記憶物理刪除）
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

### 記憶生命週期引擎

系統採用**事件驅動**策略管理記憶的生命週期 — 所有觸發邏輯掛在已有的 request path 上，不需要外部排程器（Celery、cron 等）。

#### 設計理念

參考 LightMem 的 Atkinson-Shiffrin 認知模型，記憶的遷移和清理由三個自然觸發點驅動：

```
record_message ── 系統心跳 ──────────────────────────
  ├─ append to L1
  └─ 每 N 則 → buffer drain → extraction → L2/L3

end_session ── 最終清理 ──────────────────────────────
  ├─ extract 殘餘訊息（上次 drain 之後的）→ L2/L3
  ├─ session summary → episodic
  ├─ consolidation check → background thread
  │   └─ cluster → merge → promote → dedup
  └─ lazy cleanup → 物理刪除過期資料

start_session ── 孤兒回收 ────────────────────────────
  └─ 清理 > N 小時未活動的其他 session working messages
```

#### 觸發機制一覽

| 觸發點 | 時機 | 動作 | 阻塞？ |
|--------|------|------|--------|
| **Buffer Drain** | `record_message` 每 `buffer_drain_every` 則（預設 8） | 對新訊息執行 sensory → extraction → conflict resolution → 寫入 L2/L3 | 是（同步） |
| **殘餘 Extraction** | `end_session` | 對上次 drain 之後的殘餘訊息執行 extraction | 是（同步） |
| **Consolidation** | `end_session`，當新 canonical 記憶 >= `consolidation_trigger_threshold`（預設 20） | cluster + merge + staleness detection + episodic→semantic promotion | 否（background thread） |
| **Lazy Cleanup** | `end_session` | 物理刪除 superseded/deleted > N 天的 canonical 記憶 | 是（輕量 SQL） |
| **孤兒回收** | `start_session` | 刪除該 user 超過 N 小時未活動的 working messages | 是（輕量 SQL） |

#### Buffer Drain 詳解

`record_message` 不再只是 append — 它是整個系統的心跳。

```
訊息 1  → append to L1
訊息 2  → append to L1
...
訊息 8  → append to L1 + 觸發 drain
            ↓
         取出訊息 1~8
         sensory filter → LLM extraction → conflict resolution
         寫入 L2 (event) / L3 (fact/preference/decision)
         記錄 drain offset = 8
訊息 9  → append to L1
...
訊息 16 → append to L1 + 觸發 drain
            ↓
         取出訊息 9~16（只處理新的）
         ...寫入 L2/L3
         更新 drain offset = 16
...
session/end → 取出訊息 17~N（殘餘）→ extraction → 清除 L1
```

**重複 extraction 的處理**：drain offset 追蹤在記憶體中（process-level）。若 process restart 導致 offset 遺失，conflict resolution 的 NOOP action 會自動去重，不會造成重複記憶。

#### Consolidation 觸發條件

不使用定時排程。在 `end_session` 時 piggyback 檢查：

1. 查詢 `consolidation_log` 取得上次成功 consolidation 的時間
2. 計算之後新增的 active canonical 記憶數
3. 若 >= `consolidation_trigger_threshold`（預設 20），啟動 background thread

Background thread 使用**獨立的 DB connection**（不佔用 request connection），並透過 `pg_try_advisory_lock` 確保同一 user 不會同時執行多個 consolidation。執行：
- `MemoryConsolidator.consolidate()` — 叢集合併 + 腐化檢測
- `promote_events_to_semantic()` — 老舊 event（> 90 天）晉升為 semantic fact

#### Lazy Cleanup 範圍

| 清理目標 | 條件 | 上限 |
|----------|------|------|
| 孤兒 working messages | `created_at < NOW() - orphan_session_timeout_hours` | start_session 時清理 |
| superseded canonical memories | `status IN ('superseded','deleted') AND updated_at < NOW() - superseded_cleanup_days` | end_session 時，每次最多清理一批 |

#### 相關設定參數

| 參數 | 環境變數 | 預設值 | 說明 |
|------|---------|--------|------|
| `buffer_drain_every` | `OPENCLAW_BUFFER_DRAIN_EVERY` | 8 | 每 N 則訊息觸發一次 mid-session extraction |
| `consolidation_trigger_threshold` | `OPENCLAW_CONSOLIDATION_TRIGGER_THRESHOLD` | 20 | 累積 N 筆新 canonical 記憶後觸發 consolidation |
| `orphan_session_timeout_hours` | `OPENCLAW_ORPHAN_SESSION_TIMEOUT_HOURS` | 2.0 | 超過 N 小時未活動的 session 視為孤兒 |
| `superseded_cleanup_days` | `OPENCLAW_SUPERSEDED_CLEANUP_DAYS` | 30 | superseded/deleted 記憶保留天數 |

---

## Token 監控與預算

系統內建三層 token 監控機制，從單次呼叫追蹤到每日預算控管。

### 架構總覽

```
LLM 呼叫
  │
  ▼
TokenTracker（utils/llm.py）── 呼叫層
  ├─ 攔截每次 LLM 呼叫
  ├─ 記錄 input/output tokens、耗時、operation
  └─ 累計後交給 DB 層寫入
          │
          ▼
user_profiles（db/queries.py）── 持久化層
  ├─ tokens_used_today — 當日累計用量
  ├─ token_budget_daily — 每日上限（預設 100,000）
  ├─ budget_reset_at — 跨日自動重置
  └─ SELECT ... FOR UPDATE — 原子檢查防併發超額
          │
          ▼
Admin API（routers/admin.py）── 管理層
  ├─ GET  /admin/usage/{user_id} — 查詢用量
  └─ POST /admin/budget/{user_id} — 設定預算
```

### TokenTracker — 呼叫層追蹤

`TokenTracker`（`utils/llm.py`）是一個透明包裝器，攔截 `Callable[[str], str]` 的 LLM 呼叫。

```python
class TokenTracker:
    def __init__(self, llm_fn, *, model="gpt-4o-mini", operation="")
    def __call__(self, prompt: str) -> str        # 透明代理，記錄 tokens
    def with_operation(self, operation: str)       # 共用 call log 但標記不同 operation
    def summary(self) -> dict                      # 按 operation 分組的用量摘要
    def reset(self)                                # 清除記錄

    # Properties
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    call_count: int
```

**哪些操作會被追蹤：**

| Operation | 觸發位置 | 說明 |
|-----------|---------|------|
| `extraction` | `_extract_and_store` | LLM 記憶萃取 |
| `conflict` | `_store_single` | 衝突解析（CRUD 判斷） |
| `rerank` | `search` | LLM 重排序（可選） |
| `answer` | `answer` | RAG 答案生成 |
| `consolidation` | `_maybe_consolidate` | 背景合併（background thread） |
| `promotion` | `_maybe_consolidate` | Episodic → Semantic 晉升 |

每個 operation 可以配置獨立的 LLM model（透過 `OPENCLAW_{OP}_LLM_MODEL` 環境變數）。

### 預算控管流程

```mermaid
flowchart LR
    REQ["LLM 操作請求"] --> CHK{"_check_budget<br/>SELECT ... FOR UPDATE"}
    CHK -- "remaining > 0" --> EXEC["執行 LLM 呼叫<br/>（TokenTracker 追蹤）"]
    CHK -- "remaining <= 0" --> SKIP["跳過 LLM<br/>回傳 budget_exceeded"]
    EXEC --> REC["_record_usage<br/>原子遞增 tokens_used_today"]
    REC --> DONE["完成"]

    style SKIP fill:#f66,color:#fff
```

**預算檢查點：**
- `_extract_and_store` — extraction + conflict 之前檢查，超額時跳過整個 ingest pipeline
- `answer` — RAG 生成之前檢查，超額時回傳 `abstain=True, abstain_reason="Daily token budget exceeded"`

**每日自動重置**：`check_token_budget` 檢查 `budget_reset_at::date < now()::date`，跨日時自動歸零 `tokens_used_today`。

**併發安全**：`check_token_budget` 使用 `SELECT ... FOR UPDATE` 鎖定 row，確保多個併發請求不會同時通過預算檢查後超額消耗。

### Admin API

```bash
# 查詢用量
curl http://localhost:8000/admin/usage/user_123
# 回傳：{"user_id": "user_123", "budget": 100000, "used": 12345, "remaining": 87655, "reset_at": "...", "updated_at": "..."}

# 設定每日預算
curl -X POST http://localhost:8000/admin/budget/user_123 \
  -H "Content-Type: application/json" \
  -d '{"daily_limit": 200000}'
```

---

## 三層記憶模型

| 層級 | 名稱 | 資料表 | 特性 | 用途 |
|------|------|--------|------|------|
| **L1** | Working Memory | `working_messages` | Session 範圍，最多 20 筆，用完即清 | 當前對話上下文 |
| **L2** | Episodic Memory | `episodic_memories` | 時間衰減（半衰期 30 天），event 類型為主 | 近期事件記錄 |
| **L3** | Semantic Memory | `semantic_memories` + `canonical_memories` | 永久儲存，不衰減 | 長期事實、偏好、決策 |

### 每層存了什麼 & 搜尋能力

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L1  working_messages                                                        │
│                                                                              │
│ 存了什麼：raw text（原始對話）                                               │
│ ┌────────────────────────────────────────────────────────────┐               │
│ │ id | user_id | session_id | role | content     | created_at│               │
│ │ ...│ user_1  │ sess_abc   │ user │ "我喜歡拉麵" │ 09:15     │               │
│ └────────────────────────────────────────────────────────────┘               │
│                                                                              │
│ ❌ 無 embedding       ❌ 無 tsvector         ❌ 不參與常規搜尋               │
│ ✅ 可透過 search(include_working=True) 以 substring match 加入結果           │
│ ✅ session/end 時被 extraction pipeline 消化後刪除                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ L2  episodic_memories                                                        │
│                                                                              │
│ 存了什麼：LLM 萃取後的結構化記憶 + embedding + 自動 tsvector                 │
│ ┌──────────────────────────────────────────────────────────────────────┐     │
│ │ id | user_id | content              | embedding    | memory_type     │     │
│ │ ...│ user_1  │ "使用者喜歡吃拉麵"    │ [0.12, ...]  │ event           │     │
│ │ ...│ user_1  │ "Session摘要: ..."    │ [0.08, ...]  │ session         │     │
│ └──────────────────────────────────────────────────────────────────────┘     │
│ + tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content))        │
│                                                                              │
│ ✅ embedding (vector cosine search)   ← HNSW 索引                           │
│ ✅ tsvector (keyword BM25 search)     ← GIN 索引                            │
│ ✅ 搜尋時被查詢（search_vector + search_keyword 都掃這張表）                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ L3  semantic_memories                                                        │
│                                                                              │
│ 存了什麼：LLM 萃取後的結構化記憶 + embedding + 自動 tsvector                 │
│ ┌──────────────────────────────────────────────────────────────────────┐     │
│ │ id | user_id | content               | embedding   | memory_type     │     │
│ │ ...│ user_1  │ "使用者偏好拉麵"       │ [0.15, ...] │ preference      │     │
│ │ ...│ user_1  │ "使用者住在台北"       │ [0.22, ...] │ fact            │     │
│ └──────────────────────────────────────────────────────────────────────┘     │
│ + tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content))        │
│                                                                              │
│ ✅ embedding (vector cosine search)   ← HNSW 索引                           │
│ ✅ tsvector (keyword BM25 search)     ← GIN 索引                            │
│ ✅ 搜尋時被查詢                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ L3  canonical_memories（conflict resolution 產物）                            │
│                                                                              │
│ 存了什麼：去重後的唯一事實 + embedding（無自動 tsvector，即時計算）            │
│ ┌──────────────────────────────────────────────────────────────────────┐     │
│ │ id | user_id | memory_key        | value         | embedding | status│     │
│ │ ...│ user_1  │ "user_food_pref"  │ "喜歡拉麵"    │ [0.15,..] │ active│     │
│ │ ...│ user_1  │ "user_location"   │ "住在台北"    │ [0.22,..] │ active│     │
│ └──────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│ ✅ embedding (vector cosine search)   ← HNSW 索引                           │
│ ✅ keyword search（即時 to_tsvector('english', value)，無預建 GIN 索引）      │
│ ✅ 搜尋時被查詢                                                              │
│ ⚠️  keyword search 較慢（每次現算 tsvector，不像 L2/L3 有 STORED column）     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 搜尋涵蓋範圍一覽

| 搜尋方式 | L1 working | L2 episodic | L3 semantic | canonical |
|----------|:----------:|:-----------:|:-----------:|:---------:|
| **Vector（cosine）** | ❌ | ✅ HNSW 索引 | ✅ HNSW 索引 | ✅ HNSW 索引 |
| **Keyword（BM25）** | ❌ | ✅ GIN 索引（STORED tsv） | ✅ GIN 索引（STORED tsv） | ✅ 即時計算（無 GIN） |
| **Hybrid merge** | ❌ | ✅ vec×0.7 + text×0.3 | ✅ vec×0.7 + text×0.3 | ✅ vec×0.7 + text×0.3 |
| **Temporal decay** | — | ✅ 半衰期 30天 | ✅ 半衰期 30天 | ✅ 半衰期 30天 |
| **include_working** | ✅ 直接回傳 raw | — | — | — |

### Hybrid Search 在每層的運作流程

一次搜尋實際上是**三張表各跑兩次查詢**，然後全部混在一起排序。

```
使用者查詢: "拉麵"
       │
       ├──→ embedding_provider.embed_query("拉麵") → query_vec [0.12, 0.08, ...]
       │
       │    ┌─────────────────────────────────────────────────────────────────┐
       │    │  for tbl in [episodic, semantic, canonical]:                    │
       │    │                                                                 │
       │    │    ① search_vector(query_vec, table=tbl)                        │
       │    │       SQL: SELECT ... 1.0 - (embedding <=> query_vec) AS sim    │
       │    │            FROM {tbl} WHERE user_id = ? AND embedding IS NOT NULL│
       │    │            ORDER BY embedding <=> query_vec LIMIT 20            │
       │    │                                                                 │
       │    │    ② search_keyword("拉麵", table=tbl)                          │
       │    │       SQL: SELECT ... ts_rank(tsv_expr, plainto_tsquery(?))     │
       │    │            FROM {tbl} WHERE user_id = ?                          │
       │    │            AND tsv_expr @@ plainto_tsquery(?)                    │
       │    │            ORDER BY rank DESC LIMIT 20                           │
       │    └─────────────────────────────────────────────────────────────────┘
       │
       │    三張表 × 兩種搜尋 = 最多 6 次 SQL 查詢
       │
       ▼
  merge_hybrid_results()
       │
       │  同一個 memory id 如果同時出現在 vector 和 keyword 結果：
       │    score = 0.7 × vector_score + 0.3 × text_score
       │
       │  只出現在 vector：score = 0.7 × vector_score + 0
       │  只出現在 keyword：score = 0 + 0.3 × text_score
       │
       ▼
  apply_ranking_pipeline()
       │
       ├─ Temporal Decay: score × 2^(-age_days / 30)
       ├─ MMR: 去除語義重複（λ=0.7 平衡 relevance vs diversity）
       └─ LLM Rerank（可選）
       │
       ▼
  Top-K 結果（混合了三張表的記憶，統一排序）
```

**各表 Keyword Search 的差異：**

| 表格 | tsv 來源 | 索引 | 效能 |
|------|---------|------|------|
| `episodic_memories` | `tsv` STORED column（`to_tsvector('english', content)`，寫入時自動計算） | GIN 索引 | ✅ 快（索引查找） |
| `semantic_memories` | `tsv` STORED column（同上） | GIN 索引 | ✅ 快（索引查找） |
| `canonical_memories` | 即時計算 `to_tsvector('english', value)`（無預建欄位） | ❌ 無 GIN | ⚠️ 慢（全表掃描 + 即時計算） |

> **為什麼 canonical 沒有 STORED tsvector？**
> canonical_memories 的文字欄位叫 `value`（不是 `content`），且 `value` 會被 conflict resolution 頻繁 UPDATE。
> STORED generated column 在每次 UPDATE 時都要重算，目前設計選擇用即時計算避免寫入開銷。
> 如果 canonical 數量成長到查詢變慢，可以加上 `tsv` STORED column + GIN 索引。

### 如何驗證每層有資料

```bash
# === L1: Working Memory（session 進行中才有） ===
# 查看某 session 的 working messages
psql -c "SELECT id, role, left(content, 50), created_at
         FROM working_messages
         WHERE user_id = 'USER' AND session_id = 'SESS'
         ORDER BY created_at;"

# === L2: Episodic Memory ===
# 查看最近的 episodic 記憶，確認有 embedding
psql -c "SELECT id, memory_type, left(content, 60),
                CASE WHEN embedding IS NOT NULL THEN '✅' ELSE '❌' END AS has_emb,
                created_at
         FROM episodic_memories
         WHERE user_id = 'USER'
         ORDER BY created_at DESC LIMIT 10;"

# === L3: Semantic Memory ===
psql -c "SELECT id, memory_type, left(content, 60),
                CASE WHEN embedding IS NOT NULL THEN '✅' ELSE '❌' END AS has_emb,
                created_at
         FROM semantic_memories
         WHERE user_id = 'USER'
         ORDER BY created_at DESC LIMIT 10;"

# === Canonical Memory（L3 去重後） ===
psql -c "SELECT id, memory_key, left(value, 60), status, confidence,
                CASE WHEN embedding IS NOT NULL THEN '✅' ELSE '❌' END AS has_emb,
                created_at
         FROM canonical_memories
         WHERE user_id = 'USER' AND status = 'active'
         ORDER BY created_at DESC LIMIT 10;"

# === 各層筆數統計 ===
psql -c "SELECT
           (SELECT COUNT(*) FROM working_messages WHERE user_id = 'USER') AS l1_working,
           (SELECT COUNT(*) FROM episodic_memories WHERE user_id = 'USER') AS l2_episodic,
           (SELECT COUNT(*) FROM semantic_memories WHERE user_id = 'USER') AS l3_semantic,
           (SELECT COUNT(*) FROM canonical_memories WHERE user_id = 'USER' AND status = 'active') AS canonical_active;"

# === 透過 API 查看 ===
# 列出所有記憶（跨 L2/L3/canonical，不含 L1）
curl "http://localhost:8000/memory/USER?limit=20"

# Tier 1 搜尋（確認各層都有結果回來）
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"user_id": "USER", "query": "拉麵", "limit": 20}'
```

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
   - Token 預算以 `SELECT ... FOR UPDATE` 原子檢查，防止併發超額

---

## 模組詳解

### `core/service.py` — MemoryService（新版精簡）

主要協調器，所有高層操作的入口點。

```python
class MemoryService:
    def __init__(self, embedding_provider, settings, llm_fn=None)

    # Session 生命週期
    def start_session(self, conn, user_id, session_id)        # + 孤兒 session 清理
    def record_message(self, conn, user_id, session_id, role, content)  # + buffer drain
    def end_session(self, conn, user_id, session_id) -> dict   # + consolidation + cleanup

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

| Provider | 預設模型 | 維度 | 環境變數 |
|----------|---------|------|----------|
| `openai` | text-embedding-3-small | 1536 | `OPENAI_API_KEY` |
| `gemini` | gemini-embedding-001 | 768 | `GEMINI_API_KEY` 或 `GOOGLE_API_KEY` |
| `voyage` | voyage-4-large | 1024 | `VOYAGE_API_KEY` |
| `local` | llama-cpp GGUF | 依模型 | 需要 `llama-cpp-python` |

**向量維度設定**：`embedding_dimensions` 必須匹配所選模型的輸出維度。若模型輸出維度不同，系統會自動截斷（適用於支援 Matryoshka/MRL 的模型如 OpenAI `text-embedding-3-*`）或零填充。變更維度需要重建 schema（DROP + CREATE 表格或新建資料庫）。

### `memory/working.py` — L1 Working Memory

```python
class WorkingMemory:
    def __init__(max_messages=20)
    def append(conn, user_id, session_id, role, content)
    def count(conn, user_id, session_id) -> int               # buffer drain 用
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
        vector embedding "維度由 embedding_dimensions 設定"
        text memory_type
        timestamptz created_at
        jsonb metadata
        tsvector tsv "auto-generated"
    }

    semantic_memories {
        uuid id PK
        text user_id
        text content
        vector embedding "維度由 embedding_dimensions 設定"
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
        vector embedding "維度由 embedding_dimensions 設定"
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
        int token_budget_daily "預設 100000"
        int tokens_used_today "每日自動重置"
        timestamptz budget_reset_at
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

| Method | Path | Request | Response | 說明 |
|--------|------|---------|----------|------|
| `POST` | `/search/answer` | `{user_id, query, top_k?, session_id?}` | `{answer, confidence, abstain, evidence[]}` | **★ Chatbot 主路徑** — hybrid search → RAG 一步到位 |
| `POST` | `/search` | `{user_id, query, limit?}` | `{results[{id, title, type, score}]}` | 預留 Tier 1 — 輕量索引 |
| `GET` | `/search/timeline/{memory_id}?user_id=&depth_before=3&depth_after=3` | — | `{id, content, neighbors[]}` | 預留 Tier 2 — 時序上下文 |
| `POST` | `/search/detail` | `{memory_ids[]}` | `{results[{id, content, metadata}]}` | 預留 Tier 3 — 完整內容 |

### Admin

| Method | Path | Request | Response |
|--------|------|---------|----------|
| `GET` | `/admin/usage/{user_id}` | — | `{user_id, budget, used, remaining, reset_at, updated_at}` |
| `POST` | `/admin/budget/{user_id}` | `{daily_limit}` | `{user_id, daily_limit}` |

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
export OPENCLAW_EMBEDDING_DIMENSIONS=1536  # 需匹配模型輸出維度

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
2. 設定 Embedding 向量維度（`embedding_dimensions`，寫入 schema DDL）
3. 執行 Schema Migration（idempotent，安全重複執行）
4. 建立 Embedding Provider
5. 初始化 MemoryService 單例

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
| `embedding_dimensions` | `OPENCLAW_EMBEDDING_DIMENSIONS` | 1536 | 向量維度（需匹配模型輸出，變更需 schema migration） |

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

### 記憶生命週期

| 參數 | 環境變數 | 預設值 | 說明 |
|------|---------|--------|------|
| `buffer_drain_every` | `OPENCLAW_BUFFER_DRAIN_EVERY` | 8 | 每 N 則訊息觸發一次 mid-session extraction |
| `consolidation_trigger_threshold` | `OPENCLAW_CONSOLIDATION_TRIGGER_THRESHOLD` | 20 | 累積 N 筆新 canonical 記憶後觸發 consolidation |
| `orphan_session_timeout_hours` | `OPENCLAW_ORPHAN_SESSION_TIMEOUT_HOURS` | 2.0 | 孤兒 session 超時（小時） |
| `superseded_cleanup_days` | `OPENCLAW_SUPERSEDED_CLEANUP_DAYS` | 30 | superseded/deleted 記憶物理刪除天數 |
