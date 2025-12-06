"""
Vectorstore management for NTDS Complication Extractor.
Handles text chunking and ChromaDB vectorstore creation/loading.
"""

import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def vectorstore_exists(persist_dir: str) -> bool:
    """Check if ChromaDB directory exists and is valid.

    Args:
        persist_dir: Path to the ChromaDB persistence directory

    Returns:
        bool: True if valid ChromaDB exists, False otherwise
    """
    if not os.path.exists(persist_dir) or not os.path.isdir(persist_dir):
        return False
    # Verify it's a valid ChromaDB by checking for sqlite file
    return os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))


def create_vectorstore(
    notes_df: pd.DataFrame,
    embeddings,
    persist_dir: str,
    chunk_size: int,
    chunk_overlap: int
) -> Chroma:
    """Create new ChromaDB vectorstore from all notes with chunking.

    Args:
        notes_df: DataFrame with 'note_text' column and encounter_id index
        embeddings: Embedding function (e.g., OllamaEmbeddings)
        persist_dir: Directory to persist the vectorstore
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Chroma: Initialized and populated vectorstore
    """
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    # Initialize ChromaDB with persistence
    vectorstore = Chroma(
        collection_name="ntds_notes",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # Process all notes
    all_chunks = []
    all_metadatas = []

    print(f"Processing {len(notes_df)} notes...")
    for idx, (encounter_id, row) in enumerate(notes_df.iterrows()):
        note_text = row['note_text']

        # Chunk the note
        chunks = splitter.split_text(note_text)

        # Create metadata for each chunk
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                'encounter_id': str(encounter_id),
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks)
            })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(notes_df)} notes...")

    print(f"\nAdding {len(all_chunks)} chunks to ChromaDB...")
    vectorstore.add_texts(texts=all_chunks, metadatas=all_metadatas)

    print(f"  Successfully created ChromaDB with {len(all_chunks)} chunks")
    print(f"  Persisted to: {persist_dir}")

    return vectorstore


def load_vectorstore(embeddings, persist_dir: str) -> Chroma:
    """Load existing ChromaDB vectorstore.

    Args:
        embeddings: Embedding function (must match the one used during creation)
        persist_dir: Directory where the vectorstore is persisted

    Returns:
        Chroma: Loaded vectorstore
    """
    vectorstore = Chroma(
        collection_name="ntds_notes",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore


def get_or_create_vectorstore(
    notes_df: pd.DataFrame,
    embeddings,
    persist_dir: str,
    config: dict
) -> Chroma:
    """Get existing vectorstore or create a new one if it doesn't exist.

    Args:
        notes_df: DataFrame with all notes
        embeddings: Embedding function
        persist_dir: Directory for vectorstore persistence
        config: Configuration dict with chunk_size and chunk_overlap

    Returns:
        Chroma: Loaded or newly created vectorstore
    """
    if vectorstore_exists(persist_dir):
        print(f" Found existing embeddings at {persist_dir}")
        return load_vectorstore(embeddings, persist_dir)
    else:
        print(f"Creating new vectorstore at {persist_dir}...")
        return create_vectorstore(
            notes_df,
            embeddings,
            persist_dir,
            config["chunk_size"],
            config["chunk_overlap"]
        )
