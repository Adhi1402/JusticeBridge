"""
Thin CLI to (re)build the Chroma index from data/corpus.json.

Run:  python -m justicebridge.build_index
"""

from .retrieval import build_index

if __name__ == "__main__":
    build_index()
