# Vector Store Module
# 向量化文档检索模块
from .base import (
    DocumentChunk,
    SearchResult,
    SearchQuery,
    SearchResponse,
    SearchMode,
    VectorStoreType
)
from .numpy_store import NumpyVectorStore
from .qdrant_store import QdrantVectorStore
from .milvus_store import MilvusVectorStore

__all__ = [
    # 基础数据结构
    "DocumentChunk",
    "SearchResult",
    "SearchQuery",
    "SearchResponse",
    "SearchMode",
    "VectorStoreType",
    # 存储实现
    "NumpyVectorStore",
    "QdrantVectorStore",
    "MilvusVectorStore",
]


# ================= 工厂函数 =================

def create_vector_store(
    store_type: str = "numpy",
    **kwargs
):
    """
    创建向量存储的工厂函数

    Args:
        store_type: 存储类型 ("numpy", "qdrant", "milvus")
        **kwargs: 传递给具体存储类的参数

    Returns:
        向量存储实例
    """
    if store_type == "numpy":
        return NumpyVectorStore(**kwargs)
    elif store_type == "qdrant":
        return QdrantVectorStore(**kwargs)
    elif store_type == "milvus":
        return MilvusVectorStore(**kwargs)
    else:
        raise ValueError(f"未知的存储类型: {store_type}")
