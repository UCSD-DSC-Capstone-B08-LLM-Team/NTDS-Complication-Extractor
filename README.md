# NTDS-Complication-Extractor
This repository creates a baseline RAG pipeline to test the ability of an LLM to summarize medical notes in the NTDS-18 Benchmark. The NTDS-18 benchmark is used by TQIP to evaluate hospital performance.

The notes generated in data/syntheic_ntds_trauma_notes_gemini.csv are synthetic notes. These were generated with the help of an LLM. data/ntds_18_complications.json provides an overview of all the complications tested for within the NTDS-18 dataset.

## Quickstart
1. Environment set up
To set up the conda environment, type
```
conda env create -f environment.yml
```

2. Install llama3-1:8b. First, download ollama if you have not done that yet. Then, run:
```
ollama pull llama3.1:8b
```
NOTE: You may use a different embedding model if you don't want to download this. Specify the information in the config file
3. Set up gemini API key to use a model. You can use the free API from [here](https://ai.google.dev/gemini-api/docs/api-key) to generate this API key. After generating the key, create a .env in the root and store the API key in there.

4. Configure JSON file (config.json) as intended

5. Run python file
```
python src/extraction.py
```
You can also specify a custom config file path:
```
python src/extraction.py --config path/to/your/config.json
```
On the first run, it will take some about a minute to create the ChromaDB database.
Brief results will be shown in the console output. Detailed results will be stored in results/

## File Structure
.
├── README.md
├── config.json
├── data
│   ├── describe.txt
│   ├── ntds_18_complications.json
│   └── synthetic_ntds_trauma_notes_gemini.csv
├── environment.yml
├── notebooks
│   └── extraction.ipynb
└── src
    ├── evaluation.py
    ├── extraction.py
    └── vectorstore.py