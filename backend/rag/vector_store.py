"""Chroma 向量库：基于 chromadb 的本地持久化向量存储"""
from __future__ import annotations

import uuid
from typing import Optional

from backend.config import settings
from backend.rag.embedding import Embedder


class ChromaStore:
    """Chroma 向量库封装，懒初始化 client 与 collection"""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        collection_name: str = "paper_chunks",
    ):
        # 路径默认取配置中的 CHROMA_PATH
        self.chroma_path = str(chroma_path or settings.CHROMA_PATH)
        self.embedder = embedder or Embedder()
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _ensure(self):
        """懒初始化 client 和 collection，首次访问时建立"""
        if self._client is None:
            # 延迟导入，避免模块导入即触发 chromadb 重型初始化
            import chromadb

            self._client = chromadb.PersistentClient(path=self.chroma_path)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )

    def add_chunks(self, paper_path: str, title: str, chunks: list[dict]):
        """批量入库。

        Args:
            paper_path: 论文文件路径
            title: 论文标题
            chunks: 每个 chunk 形如 {section, text}
        """
        if not chunks:
            return
        self._ensure()
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        # 唯一 id，避免冲突
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {
                "paper_path": paper_path,
                "title": title,
                "section": c.get("section", ""),
            }
            for c in chunks
        ]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def vector_search(self, query: str, top_k: int = 5) -> list[dict]:
        """向量检索，返回 top_k 结果。

        Returns:
            list[dict]，每项 {text, paper_path, title, section, score}
            score 越大越相似（由距离转换得到）。
        """
        self._ensure()
        query_emb = self.embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )
        out: list[dict] = []
        # chroma 返回结构：每个字段是“外层 list=查询数，内层 list=结果数”
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # 距离越小越相似，转换为 (0,1] 的相似度分数
            score = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
            out.append(
                {
                    "text": doc,
                    "paper_path": meta.get("paper_path", ""),
                    "title": meta.get("title", ""),
                    "section": meta.get("section", ""),
                    "score": score,
                }
            )
        return out

    def clear(self):
        """清空 collection（删除后重建空 collection）"""
        self._ensure()
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name
        )
