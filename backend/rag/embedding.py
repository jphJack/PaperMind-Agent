"""本地 Embedding：直接用 transformers 加载 BGE 模型（绕开 sentence-transformers 与 pyarrow）。

为什么不用 sentence-transformers：
  你的环境里 pyarrow 在 import datasets 时触发 0xC0000005 访问冲突，
  而 sentence-transformers 强依赖 datasets → pyarrow，整条链路崩溃。
  这里直接用 transformers 加载 BERT backbone + CLS Pooling，向量结果与 sentence-transformers 等价。
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional

# 在 import torch 之前强制只用 CPU（避免 CUDA 不可用机器上的 0xC0000005）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 强制 transformers / huggingface_hub 走离线模式，不再请求 Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from backend.config import settings


class Embedder:
    """本地 Embedder，懒加载 BGE 系列模型。

    支持任意 HuggingFace 上有 BertModel + AutoTokenizer 结构的嵌入模型
    （bge-large-zh-v1.5 / bge-base-zh / bge-small-zh / bge-m3 等）。
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._device = "cpu"  # 强制 CPU，避免 CUDA 初始化崩溃

    def _resolve_local_snapshot(self, model_name: str) -> Optional[str]:
        """如果 HF 本地缓存已有该模型的 snapshot，直接返回快照路径。"""
        try:
            from pathlib import Path as _P
            from huggingface_hub.constants import HF_HUB_CACHE

            repo_dir_name = "models--" + model_name.replace("/", "--")
            snapshots_dir = _P(HF_HUB_CACHE) / repo_dir_name / "snapshots"
            if snapshots_dir.is_dir():
                snapshots = sorted(
                    [p for p in snapshots_dir.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if snapshots:
                    snap = snapshots[0]
                    if (snap / "config.json").exists():
                        return str(snap)
        except Exception:
            pass
        return None

    def _load(self):
        """懒加载模型：首次调用时加载，双重检查锁保证线程安全。"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    local_snapshot = self._resolve_local_snapshot(self.model_name)
                    if local_snapshot:
                        load_path = local_snapshot
                    else:
                        load_path = self.model_name

                    # 注意：bge 官方建议 query 加指令前缀 "为这个句子生成表示以用于检索中文文档："
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        load_path, local_files_only=bool(local_snapshot)
                    )
                    self._model = AutoModel.from_pretrained(
                        load_path, local_files_only=bool(local_snapshot)
                    ).to(self._device)
                    self._model.eval()

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        """BGE 系列使用的 last-token pooling（等价于 sentence-transformers 的 cls + 末尾 token）。"""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _encode(self, texts: List[str]):
        import torch

        # BGE 官方推荐：query 加指令前缀，passage 不加
        is_query = getattr(self, "_is_query", True)
        if is_query and "bge" in self.model_name.lower() and "zh" in self.model_name.lower():
            prefix = "为这个句子生成表示以用于检索中文文档："
            texts = [prefix + t for t in texts]

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            # bge 用 last_token_pool；其它 BERT 类模型退化为 [CLS] pooling
            try:
                embeddings = self._last_token_pool(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
            except Exception:
                embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量向量化（passage/documents）：不加 query 前缀。"""
        if not texts:
            return []
        self._load()
        old = getattr(self, "_is_query", True)
        self._is_query = False
        try:
            return self._encode(texts)
        finally:
            self._is_query = old

    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化：加 query 前缀。"""
        self._load()
        self._is_query = True
        return self._encode([text])[0]
