# Entity Linking Module
from .base import MatchResult, MatchStage, EntityInfo
from .fuzzy_matcher import FuzzyMatcher
from .vector_indexer import VectorIndexer
from .linker import EntityLinker, create_entity_linker, link_ner_results

__all__ = [
    "MatchResult",
    "MatchStage",
    "EntityInfo",
    "FuzzyMatcher",
    "VectorIndexer",
    "EntityLinker",
    "create_entity_linker",
    "link_ner_results",
]
