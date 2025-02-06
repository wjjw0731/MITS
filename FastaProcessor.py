import pandas as pd
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


class FastaProcessor:
    """Process FASTA files into structured DataFrame with taxonomic labels."""

    @staticmethod
    def read_fasta(file_path: str) -> tuple[list, list]:
        """Read FASTA file and extract headers/sequences."""
        headers, sequences = [], []
        current_seq = []

        with Path(file_path).open('r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                    headers.append(line[1:])  # Remove '>' prefix
                else:
                    current_seq.append(line)
            if current_seq:
                sequences.append(''.join(current_seq))
        return headers, sequences

    @staticmethod
    def parse_taxonomy(headers: list) -> pd.DataFrame:
        """Extract taxonomic labels from headers (k__...;p__... format)."""
        taxonomy_levels = ['k', 'p', 'c', 'o', 'f', 'g', 's']
        data = {level: [] for level in taxonomy_levels}

        for header in headers:
            taxa = re.split(r'[;|]', header)
            for i, level in enumerate(taxonomy_levels):
                value = taxa[i].split('__')[-1] if i < len(taxa) else None
                data[level].append(value)

        return pd.DataFrame(data)

    @classmethod
    def process(cls, file_path: str, min_samples: int = 10) -> pd.DataFrame:
        """Full processing pipeline."""
        headers, sequences = cls.read_fasta(file_path)
        df = cls.parse_taxonomy(headers)
        df['sequence'] = sequences

        # Data cleaning
        df = df[df['sequence'].str.match('^[ACGT]+$', na=False)]
        df = df[~df['s'].str.contains('Incertae', na=False)]

        # Filter rare species
        species_counts = df['s'].value_counts()
        valid_species = species_counts[species_counts >= min_samples].index
        return df[df['s'].isin(valid_species)].reset_index(drop=True)