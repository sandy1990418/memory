"""
Query expansion for FTS-only search mode.
Mirrors: src/memory/query-expansion.ts
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

_STOP_WORDS_EN: frozenset[str] = frozenset(
    [
        # Articles / determiners
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        # Pronouns
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        # Common verbs
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        # Prepositions
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        # Conjunctions
        "and",
        "or",
        "but",
        "if",
        "then",
        "because",
        "as",
        "while",
        "when",
        "where",
        "what",
        "which",
        "who",
        "how",
        "why",
        # Time refs (vague)
        "yesterday",
        "today",
        "tomorrow",
        "earlier",
        "later",
        "recently",
        "ago",
        "just",
        "now",
        # Vague references
        "thing",
        "things",
        "stuff",
        "something",
        "anything",
        "everything",
        "nothing",
        # Request words
        "please",
        "help",
        "find",
        "show",
        "get",
        "tell",
        "give",
    ]
)

_STOP_WORDS_ZH: frozenset[str] = frozenset(
    [
        "我",
        "我们",
        "你",
        "你们",
        "他",
        "她",
        "它",
        "他们",
        "这",
        "那",
        "这个",
        "那个",
        "这些",
        "那些",
        "的",
        "了",
        "着",
        "过",
        "得",
        "地",
        "吗",
        "呢",
        "吧",
        "啊",
        "呀",
        "嘛",
        "啦",
        "是",
        "有",
        "在",
        "被",
        "把",
        "给",
        "让",
        "用",
        "到",
        "去",
        "来",
        "做",
        "说",
        "看",
        "找",
        "想",
        "要",
        "能",
        "会",
        "可以",
        "和",
        "与",
        "或",
        "但",
        "但是",
        "因为",
        "所以",
        "如果",
        "虽然",
        "而",
        "也",
        "都",
        "就",
        "还",
        "又",
        "再",
        "才",
        "只",
        "之前",
        "以前",
        "之后",
        "以后",
        "刚才",
        "现在",
        "昨天",
        "今天",
        "明天",
        "最近",
        "东西",
        "事情",
        "事",
        "什么",
        "哪个",
        "哪些",
        "怎么",
        "为什么",
        "多少",
        "请",
        "帮",
        "帮忙",
        "告诉",
    ]
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _is_valid_keyword(token: str) -> bool:
    if not token:
        return False
    if re.match(r"^[a-zA-Z]+$", token) and len(token) < 3:
        return False
    if re.match(r"^\d+$", token):
        return False
    if re.match(r"^[\W_]+$", token, re.UNICODE):
        return False
    return True


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    normalized = text.lower().strip()
    segments = re.split(
        r"[\s\p{P}]+" if False else r"[\s\.,!?;:\"'()\[\]{}<>/\\|@#$%^&*=+~`]+", normalized
    )

    for segment in segments:
        if not segment:
            continue
        if _CJK_RE.search(segment):
            chars = [c for c in segment if _CJK_RE.match(c)]
            tokens.extend(chars)
            for i in range(len(chars) - 1):
                tokens.append(chars[i] + chars[i + 1])
        else:
            tokens.append(segment)

    return tokens


def extract_keywords(query: str) -> list[str]:
    """
    Extract meaningful keywords from a conversational query.
    Mirrors: query-expansion.ts::extractKeywords
    """
    tokens = _tokenize(query)
    keywords: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        if token in _STOP_WORDS_EN or token in _STOP_WORDS_ZH:
            continue
        if not _is_valid_keyword(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)

    return keywords
