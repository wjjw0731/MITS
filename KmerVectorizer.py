from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")


class KmerVectorizer:
    """Generate k-mer embeddings using pre-trained Word2Vec models."""

    _COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    def __init__(self, model_dir: str = './models/'):
        self.model_dir = Path(model_dir)
        self.models = self._load_models()

    def _load_models(self) -> Dict[int, Tuple[Word2Vec, Word2Vec]]:
        """Load forward and reverse complement models for k=7 and k=10."""
        return {
            k: (
                Word2Vec.load(str(self.model_dir / f'word2vec_for_k{k}.model')),
                Word2Vec.load(str(self.model_dir / f'word2vec_back_k{k}.model'))
            )
            for k in [7, 10]
        }

    @staticmethod
    def generate_kmers(sequence: str, k: int) -> Tuple[list, list]:
        """Generate k-mers and reverse complement k-mers."""
        kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
        rc_kmers = [''.join([KmerVectorizer._COMPLEMENT[base] for base in kmer][::-1])
                    for kmer in kmers]
        return kmers, rc_kmers

    @staticmethod
    def vectorize(kmers: list, model: Word2Vec) -> np.ndarray:
        """Convert k-mers to averaged vector."""
        vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add embedding columns to DataFrame."""
        for k in [7, 10]:
            df[[f'k{k}_kmers', f'k{k}_rc_kmers']] = df['sequence'].apply(
                lambda seq: pd.Series(self.generate_kmers(seq, k))
            )

            fw_model, rc_model = self.models[k]
            df[f'seq_vector{k}'] = df[f'k{k}_kmers'].apply(lambda x: self.vectorize(x, fw_model))
            df[f'backseq_vector{k}'] = df[f'k{k}_rc_kmers'].apply(lambda x: self.vectorize(x, rc_model))

        return df.drop(columns=[c for c in df.columns if 'kmers' in c])

    def csv2json(self, df):
        df.to_json("prediction_result.json", orient='records', lines=True)