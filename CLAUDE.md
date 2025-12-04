The goal of @extraction.ipynb is to perform a baseline RAG to detect which condition a patient has in the NTDS notes set. An LLM will be given a long note and will need to perform RAG on the specific note itself to help determine this. It should only retrieve from the specific note used.

data/synthetic_ntds_trauma_notes_gemini.csv contains all synthetic trauma notes that will be tested