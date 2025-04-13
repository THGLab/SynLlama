import pickle
from collections import Counter
from itertools import chain
from collections.abc import Iterable
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

import numpy as np
from typing import List, Tuple
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed

from synllama.chem.mol import Molecule
from synllama.chem.fpindex import _QueryResult

def string_similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def sort_by_similarity(string, target_list):
    return sorted(target_list, key=lambda x: string_similarity(string, x))

def find_closest_match(string, target_list):
    return sort_by_similarity(string, target_list)[0]

class SmilesTokenizer:
    def __init__(self, token_list_path, n_gram_range=(2, 4)):
        with open(token_list_path, "r") as f:
            self.token_list = [line.strip() for line in f.readlines()]
        self.token_list = sorted(self.token_list, key=len, reverse=True)
        self.n_gram_range = n_gram_range

    def __call__(self, smiles):
        tokens = self._tokenize(smiles)
        return self._create_ngrams(tokens)

    def _tokenize(self, smiles):
        tokens = []
        i = 0
        while i < len(smiles):
            matched = False
            for token in self.token_list:
                if smiles.startswith(token, i):
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                tokens.append(smiles[i])
                i += 1
        return tokens

    def _create_ngrams(self, tokens):
        ngrams = []
        for n in range(self.n_gram_range[0], self.n_gram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i:i+n]))
        return ngrams

def compute_embeddings(vectorizer: TfidfVectorizer, molecules: Iterable[Molecule], batch_size: int = 1024) -> np.ndarray:
    all_smiles = [mol.smiles for mol in molecules]
    
    def process_batch(batch):
        return vectorizer.transform(batch)
    
    batches = [all_smiles[i:i + batch_size] for i in range(0, len(all_smiles), batch_size)]
    
    embeddings_list = Parallel(n_jobs=-1)(
        delayed(process_batch)(batch) for batch in tqdm(batches, desc="Computing embeddings")
    )
    
    return np.vstack(embeddings_list)

class SmilesSimilaritySearch:
    def __init__(self, n_gram_range=(2, 4), token_list_path=None, max_features=1024):
        self.n_gram_range = n_gram_range
        self.token_list_path = token_list_path
        self.tokenizer = SmilesTokenizer(token_list_path, n_gram_range) if token_list_path else None
        self.ngram_vocab = None
        self.idf = None
        self._molecules = None
        self._embeddings = None
        self._tree = None
        self.max_features = max_features

    def fit(self, molecules: Iterable[Molecule], save_ngram = False):
        self._molecules = tuple(molecules)
        all_smiles = [mol.smiles for mol in self._molecules]
        
        if self.tokenizer:
            all_tokens = [self.tokenizer(smiles) for smiles in tqdm(all_smiles, desc="Tokenizing SMILES")]
        else:
            all_tokens = all_smiles

        # Generate n-gram vocabulary (now sorted and limited)
        self.ngram_vocab = self._generate_ngram_vocab(all_tokens)
        if save_ngram:
            with open("data/processed/smiles_similarity_search_ngram_vocab.txt", "w") as f:
                for ngram in self.ngram_vocab:
                    f.write(ngram + "\n")
        
        # Compute IDF (now using the limited vocabulary)
        self.idf = self._compute_idf(all_tokens)
        
        # Compute TF-IDF embeddings (now using the limited vocabulary)
        self._embeddings = self._compute_tfidf_embeddings(all_tokens)
        
        # Initialize BallTree
        self._tree = self._init_tree()

    def _generate_ngram_vocab(self, all_tokens):
        ngram_counter = Counter()
        for tokens in tqdm(all_tokens, desc="Generating n-gram vocabulary"):
            if self.tokenizer:
                ngrams = self.tokenizer._create_ngrams(tokens)
            else:
                ngrams = [tokens[i:i+n] for n in range(self.n_gram_range[0], self.n_gram_range[1]+1)
                          for i in range(len(tokens)-n+1)]
            ngram_counter.update(ngrams)
        
        # Sort n-grams by frequency and keep only the top max_features
        return [ngram for ngram, _ in ngram_counter.most_common(self.max_features)]

    def _compute_idf(self, all_tokens):
        doc_freq = Counter()
        for tokens in tqdm(all_tokens, desc="Computing IDF"):
            if self.tokenizer:
                ngrams = set(self.tokenizer._create_ngrams(tokens))
            else:
                ngrams = set(tokens[i:i+n] for n in range(self.n_gram_range[0], self.n_gram_range[1]+1)
                             for i in range(len(tokens)-n+1))
            doc_freq.update(ngram for ngram in ngrams if ngram in self.ngram_vocab)
        
        num_docs = len(all_tokens)
        return {ngram: np.log(num_docs / (count + 1)) + 1 for ngram, count in doc_freq.items()}

    def _compute_tfidf_embeddings(self, all_tokens):
        embeddings = []
        for tokens in tqdm(all_tokens, desc="Computing TF-IDF embeddings"):
            if self.tokenizer:
                ngrams = self.tokenizer._create_ngrams(tokens)
            else:
                ngrams = [tokens[i:i+n] for n in range(self.n_gram_range[0], self.n_gram_range[1]+1)
                          for i in range(len(tokens)-n+1)]
            
            tf = Counter(ngram for ngram in ngrams if ngram in self.ngram_vocab)
            tfidf = np.zeros(len(self.ngram_vocab))
            for i, ngram in enumerate(self.ngram_vocab):
                if ngram in tf:
                    tfidf[i] = tf[ngram] * self.idf.get(ngram, 0)
            embeddings.append(tfidf)
        return np.array(embeddings)

    def _init_tree(self) -> BallTree:
        return BallTree(self._embeddings, metric='manhattan')

    def query(self, query_smiles: str, k: int = 10) -> List[_QueryResult]:
        if self.tokenizer:
            query_tokens = self.tokenizer(query_smiles)
            query_ngrams = self.tokenizer._create_ngrams(query_tokens)
        else:
            query_ngrams = [query_smiles[i:i+n] for n in range(self.n_gram_range[0], self.n_gram_range[1]+1)
                            for i in range(len(query_smiles)-n+1)]
        
        tf = Counter(ngram for ngram in query_ngrams if ngram in self.ngram_vocab)
        query_embedding = np.zeros(len(self.ngram_vocab))
        for i, ngram in enumerate(self.ngram_vocab):
            if ngram in tf:
                query_embedding[i] = tf[ngram] * self.idf.get(ngram, 0)
        
        distances, indices = self._tree.query(query_embedding.reshape(1, -1), k=k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            results.append(_QueryResult(
                index=idx,
                molecule=self._molecules[idx],
                fingerprint=None,
                distance=distance
            ))
        
        return sorted(results, key=lambda x: x.distance)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
