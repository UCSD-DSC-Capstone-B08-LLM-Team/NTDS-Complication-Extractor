├── CLAUDE.md
├── README.md
├── TODO.md
├── data
│   ├── describe.txt
│   ├── ntds_18_complications.json
│   └── synthetic_ntds_trauma_notes_gemini.csv
├── environment.yml
├── extraction.ipynb
└── ntds_embeddings
    ├── c04f0a92-4d0d-4527-ac6c-c10a1fb53317
    │   ├── data_level0.bin
    │   ├── header.bin
    │   ├── length.bin
    │   └── link_lists.bin
    └── chroma.sqlite3

extraction.ipynb is the crucial file in this repository. The goal of extraction.ipynb is to perform a baseline RAG to detect which condition a patient has in the NTDS notes set. An LLM will be given a long note and will need to perform RAG on the specific note itself to help determine this. It should only retrieve from the specific note used.

data/synthetic_ntds_trauma_notes_gemini.csv contains all synthetic trauma notes that will be tested

IMPORTANT NOTES TO MODEL: Prioritize code effectiveness and quality over quantity. It is okay to take multiple steps to generate code -- but there being minimal slop and the code to work correctly matters