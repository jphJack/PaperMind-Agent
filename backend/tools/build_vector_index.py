"""步骤3 索引构建工具：将解析后的论文加入混合检索向量库"""
from __future__ import annotations

from backend.tools.base import Tool


# 模块级单例：懒加载，避免导入即触发重型依赖（Embedder/Chroma/BM25）
_hybrid_searcher = None


def _get_searcher():
    """懒加载模块级 HybridSearcher 单例。

    供 build_vector_index 与 hybrid_search_tool 共享同一实例，
    保证索引写入与检索读取状态一致。
    """
    global _hybrid_searcher
    if _hybrid_searcher is None:
        from backend.rag.hybrid_search import HybridSearcher

        _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher


async def build_vector_index(papers_json: list[dict]) -> dict:
    """将已解析论文列表加入向量库与 BM25 库。

    Args:
        papers_json: 已解析论文列表，每项含 path/title/sections

    Returns:
        {"indexed_count": N, "status": "done"}
    """
    from backend.models.schemas import ParsedPaper, Section

    searcher = _get_searcher()
    indexed = 0
    for paper_dict in papers_json:
        try:
            sections = [
                Section(
                    heading=s.get("heading", ""),
                    content=s.get("content", ""),
                    figure_captions=s.get("figure_captions", []),
                )
                for s in paper_dict.get("sections", [])
            ]
            paper = ParsedPaper(
                path=paper_dict.get("path", ""),
                title=paper_dict.get("title", ""),
                sections=sections,
            )
            await searcher.async_add_paper(paper)
            indexed += 1
        except Exception:
            # 单篇失败隔离，继续处理其余论文
            continue
    return {"indexed_count": indexed, "status": "done"}


# ---------- Tool 定义 ----------

_PARAMETERS = {
    "type": "object",
    "properties": {
        "papers_json": {
            "type": "array",
            "description": "已解析论文列表，每项含 path/title/sections",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "title": {"type": "string"},
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "heading": {"type": "string"},
                                "content": {"type": "string"},
                                "figure_captions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        }
    },
    "required": ["papers_json"],
}

build_vector_index_tool = Tool(
    name="build_vector_index",
    description=(
        "步骤3：将已解析论文列表加入向量库与 BM25 库，支持后续混合搜索。"
        "返回 {indexed_count, status}。"
    ),
    parameters=_PARAMETERS,
    func=build_vector_index,
)
