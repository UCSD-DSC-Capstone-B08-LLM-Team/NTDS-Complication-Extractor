"""
NTDS Complication Extractor - Main RAG Pipeline
Performs single-note RAG-based complication assessment for trauma patients.
"""

import sys
import json
import re
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings

from vectorstore import get_or_create_vectorstore
from evaluation import (
    compare_outputs_and_score,
    generate_evaluation_report,
    print_console_summary,
    write_results_file
)

# ============================================================================
# CONSTANTS
# ============================================================================

OUTPUT_FORMAT_TEMPLATE = """At the end of your message, Output the following as JSON:
```json
{{{{
    "{condition_name}": "<Yes or No>"
}}}}
```
"""

# ============================================================================
# CONFIGURATION AND DATA LOADING
# ============================================================================

def load_config(config_path: str = "config.json") -> dict:
    """Load and validate configuration from JSON file.

    Args:
        config_path: Path to config.json file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required fields
    required_fields = [
        "encounter_id", "complications_range", "embedding_model",
        "generation_model", "k", "chunk_size", "chunk_overlap",
        "persist_directory", "output_file"
    ]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    return config


def load_notes_data(csv_path: str) -> pd.DataFrame:
    """Load trauma notes from CSV file.

    Args:
        csv_path: Path to CSV file with notes

    Returns:
        pd.DataFrame: Notes dataframe with encounter_id as index
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Notes file not found: {csv_path}")

    notes = pd.read_csv(csv_path, index_col="encounter_id")
    return notes


def load_complications_info(json_path: str) -> list[dict]:
    """Load complications definitions from JSON file.

    Args:
        json_path: Path to complications JSON file

    Returns:
        list: List of complication definition dicts
    """
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Complications file not found: {json_path}")

    with open(json_path, 'r') as f:
        complications_info = json.load(f)

    return complications_info


def build_label_to_id_mapping(complications_info: list[dict]) -> dict:
    """Build mapping from complication labels to IDs.

    Args:
        complications_info: List of complication dicts

    Returns:
        dict: Mapping from label to id
    """
    return {complication['label']: complication['id']
            for complication in complications_info}


# ============================================================================
# RAG RETRIEVAL
# ============================================================================

def build_complication_query(complication: dict) -> str:
    """Build focused retrieval query for a single complication.

    Combines the name, definition, and clues into a query string.

    Args:
        complication: Dict with 'label', 'short_definition', 'positive_note_clues'

    Returns:
        str: Combined query string for retrieval
    """
    query_parts = [
        complication['label'],
        complication['short_definition'],
        ' '.join(complication['positive_note_clues'])
    ]
    return ' '.join(query_parts)


def retrieve_relevant_chunks(
    vectorstore,
    encounter_id: str | int,
    query: str,
    k: int
) -> list[str]:
    """Retrieve relevant chunks from vectorstore for specific encounter.

    Args:
        vectorstore: ChromaDB vectorstore
        encounter_id: Patient encounter ID
        query: Query string for retrieval
        k: Number of chunks to retrieve

    Returns:
        list: List of retrieved chunk texts
    """
    results = vectorstore.similarity_search(
        query,
        k=k,
        filter={"encounter_id": str(encounter_id)}
    )
    return [doc.page_content for doc in results]


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

def format_complication_prompt(
    complication: dict,
    retrieved_chunks: str
) -> list:
    """Format the per-complication prompt with definition and retrieved chunks.

    Args:
        complication: Dict with 'label', 'short_definition', 'positive_note_clues'
        retrieved_chunks: String containing retrieved and formatted chunks

    Returns:
        list: Formatted messages ready for LLM
    """
    output_instructions = OUTPUT_FORMAT_TEMPLATE.format(
        condition_name=complication['label']
    )

    # Format positive clues
    clues_text = "\n".join([
        f"  {i+1}. {clue}"
        for i, clue in enumerate(complication['positive_note_clues'])
    ])

    # Build system message
    system_message = f"""You are an expert registrar who is highly experienced at meeting the
National Trauma Data Standard (NTDS). You are analyzing a patient's medical note from
UCSD Health, a level 1 trauma center. Your task is to determine whether this patient
has a SPECIFIC complication. Go step by step and state your train of thought.

COMPLICATION TO ASSESS:
- Name: {complication['label']}
- Definition: {complication['short_definition']}
- Positive Indicators (what to look for):
{clues_text}

{output_instructions}"""

    per_complication_template = ChatPromptTemplate([
        ("system", system_message),
        ("human", "{retrieved_chunks}"),
        ("system", "Based on the retrieved chunks above, provide your assessment.")
    ])

    return per_complication_template.format_messages(
        retrieved_chunks=retrieved_chunks
    )


def parse_llm_response(response_content: str, complication_label: str) -> bool:
    """Extract Yes/No from LLM JSON response.

    Args:
        response_content: Raw LLM response string
        complication_label: Label of the complication being assessed

    Returns:
        bool: True if Yes, False if No

    Raises:
        ValueError: If response format is invalid
    """
    json_regex = r"```json\n(.*?)\s*```"
    match = re.search(json_regex, response_content, flags=re.DOTALL)

    if match is None:
        raise ValueError(f"No JSON match found in LLM response for {complication_label}")

    json_content = match.group(1)
    parsed = json.loads(json_content)
    response = parsed[complication_label]

    if response.casefold() == "no":
        return False
    elif response.casefold() == "yes":
        return True
    else:
        raise ValueError(
            f"Invalid response for {complication_label}: '{response}' (expected 'Yes' or 'No')"
        )


# ============================================================================
# ASSESSMENT
# ============================================================================

def assess_single_complication(
    complication: dict,
    encounter_id: str | int,
    vectorstore,
    llm,
    k: int = 2,
    verbose: bool = False
) -> tuple[str, bool]:
    """Assess whether a patient has a specific complication using RAG.

    Args:
        complication: Dict with 'id', 'label', 'short_definition', 'positive_note_clues'
        encounter_id: Patient encounter ID
        vectorstore: ChromaDB vectorstore
        llm: LLM instance (e.g., ChatGoogleGenerativeAI)
        k: Number of chunks to retrieve
        verbose: Whether to print detailed progress

    Returns:
        tuple: (complication_label, predicted_bool)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Assessing: {complication['label']} (ID: {complication['id']})")
        print(f"{'='*70}")

    # 1. Build focused query
    query = build_complication_query(complication)
    if verbose:
        print(f"\n1. Built query")

    # 2. Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(vectorstore, encounter_id, query, k=k)
    if verbose:
        print(f"\n2. Retrieved {len(chunks)} chunks")

    # 3. Format context
    retrieved_context = "\n\n---\n\n".join([
        f"Relevant Section {i+1}:\n{chunk}"
        for i, chunk in enumerate(chunks)
    ])
    if verbose:
        print(f"\n3. Formatted context (length: {len(retrieved_context)} chars)")

    # 4. Format messages
    messages = format_complication_prompt(complication, retrieved_context)
    if verbose:
        print(f"\n4. Prepared LLM prompt")

    # 5. Call LLM
    if verbose:
        print("\n5. Calling LLM...")
    response = llm.invoke(messages)

    # 6. Parse response
    prediction = parse_llm_response(response.content, complication['label'])
    if verbose:
        print(f"\n6. LLM Response: {prediction}")

    return complication['label'], prediction


def assess_all_complications(
    encounter_id: str | int,
    complications_info: list[dict],
    complications_range: tuple[int, int],
    vectorstore,
    llm,
    k: int = 2,
    verbose: bool = False
) -> dict:
    """Assess all complications in specified range for an encounter.

    Args:
        encounter_id: Patient encounter ID
        complications_info: List of all complication dicts
        complications_range: (start, end) tuple for range of complications
        vectorstore: ChromaDB vectorstore
        llm: LLM instance
        k: Number of chunks to retrieve per complication
        verbose: Whether to print detailed progress

    Returns:
        dict: Mapping from complication labels to predicted bools
    """
    start_idx, end_idx = complications_range
    outputs_dict = {}

    print("\n" + "=" * 50)
    print("NTDS COMPLICATION EXTRACTOR")
    print("=" * 50)
    print(f"\nAssessing complications for encounter {encounter_id}...")
    print(f"Complications range: [{start_idx}, {end_idx})")

    for i in range(start_idx, end_idx):
        complication = complications_info[i]
        complication_label = complication['label']

        label, prediction = assess_single_complication(
            complication=complication,
            encounter_id=encounter_id,
            vectorstore=vectorstore,
            llm=llm,
            k=k,
            verbose=verbose
        )

        outputs_dict[label] = prediction

        # Print progress
        status = "Yes" if prediction else "No"
        print(f"  [{i+1-start_idx}/{end_idx-start_idx}] {complication_label}: {status}")

    return outputs_dict


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    try:
        # 0. Parse command-line arguments
        parser = argparse.ArgumentParser(
            description="NTDS Complication Extractor - RAG-based complication assessment"
        )
        parser.add_argument(
            "--config",
            type=str,
            default="config.json",
            help="Path to configuration JSON file (default: config.json)"
        )
        args = parser.parse_args()

        # 1. Load config
        print("Loading configuration...")
        config = load_config(args.config)

        # 2. Initialize environment
        print("Initializing environment...")
        load_dotenv()

        embeddings = OllamaEmbeddings(model=config["embedding_model"])
        llm = ChatGoogleGenerativeAI(model=config["generation_model"])

        # 3. Load data
        print("\nLoading data...")

        # Build paths
        data_dir = Path(config.get("data_dir", "./data"))
        notes_path = data_dir / config.get("notes_file", "synthetic_ntds_trauma_notes_gemini.csv")
        complications_path = data_dir / config.get("complications_file", "ntds_18_complications.json")

        notes_df = load_notes_data(str(notes_path))
        complications_info = load_complications_info(str(complications_path))
        mapping_dict = build_label_to_id_mapping(complications_info)

        print(f"Loaded {len(notes_df)} trauma notes")
        print(f"Loaded {len(complications_info)} complication definitions")

        # Get ground truth for encounter
        encounter_id = config["encounter_id"]
        if encounter_id not in notes_df.index:
            raise ValueError(f"Encounter ID {encounter_id} not found in dataset")

        # Extract ground truth (complications columns start at index 8)
        FIRST_COMPLICATION_COL = 8
        complication_cols = notes_df.columns[FIRST_COMPLICATION_COL:]
        ground_truth = {col: bool(notes_df.loc[encounter_id, col]) for col in complication_cols}

        # 4. Setup vectorstore (skip if exists)
        print("\nSetting up vectorstore...")
        vectorstore = get_or_create_vectorstore(
            notes_df,
            embeddings,
            config["persist_directory"],
            config
        )

        # 5. Run assessment
        llm_outputs = assess_all_complications(
            encounter_id=encounter_id,
            complications_info=complications_info,
            complications_range=tuple(config["complications_range"]),
            vectorstore=vectorstore,
            llm=llm,
            k=config["k"],
            verbose=config.get("verbose", False)
        )

        # 6. Evaluate
        num_correct, num_incorrect = compare_outputs_and_score(
            llm_outputs,
            ground_truth,
            mapping_dict
        )
        evaluation_report = generate_evaluation_report(
            llm_outputs,
            ground_truth,
            mapping_dict
        )

        # 7. Output
        print_console_summary(
            llm_outputs,
            num_correct,
            num_incorrect,
            encounter_id,
            complications_info
        )

        write_results_file(
            llm_outputs,
            evaluation_report,
            config["output_file"],
            {"config": config}
        )

    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
