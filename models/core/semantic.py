# TBD
from typing import Dict, List, Tuple
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from models.utils.common import normalize_item_name, naturalize
from models.core.taxonomy import Taxonomy

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def naturalize_name(name: str) -> str:
    return naturalize(name)

class SemanticFoodMatcher:
    def __init__(self, taxonomy: Taxonomy, model_name: str = DEFAULT_MODEL, threshold: float = 0.78):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.taxonomy = taxonomy
        self._index: Dict[str, np.ndarray] = {}
        self._build_taxonomy_index()

    def _build_taxonomy_index(self):
        variants = []
        keys = []
        for sub in self.taxonomy.all_subcategories:
            item = self.taxonomy.get_item(sub)
            if not item: 
                continue
            names = {sub}
            if item.get("display_name"): names.add(item["display_name"])
            for s in item.get("similar_items", []) or []:
                if s: names.add(s)
            nat = [naturalize_name(x) for x in names]
            keys.append(sub)
            variants.append(nat)
        # average of variants per item
        flat = [t for group in variants for t in group]
        embs = self.model.encode(flat, normalize_embeddings=True)
        i = 0
        for k, group in zip(keys, variants):
            vecs = embs[i:i+len(group)]
            self._index[k] = np.mean(vecs, axis=0)
            i += len(group)

    @lru_cache(maxsize=4096)
    def embed_label(self, label: str) -> np.ndarray:
        return self.model.encode([naturalize_name(label)], normalize_embeddings=True)[0]

    def similarity(self, a: str, b: str) -> float:
        va = self.embed_label(a); vb = self.embed_label(b)
        return float(np.dot(va, vb))

    def taxonomy_similarity(self, label: str, target_sub: str) -> float:
        va = self.embed_label(label)
        vb = self._index.get(target_sub)
        if vb is None: 
            return 0.0
        return float(np.dot(va, vb))