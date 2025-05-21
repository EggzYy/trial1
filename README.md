# Document Translation and Analysis Pipeline

This project provides a Python script (`trial_integrated4.py`) for translating documents from a source language to a target language using a pipeline of Large Language Model (LLM) agents. It also includes capabilities for document summarization and categorization. The translation process leverages the Autogen framework and is designed to work with local LLM providers like Ollama.

**Note:** This project is currently under development. Several planned enhancements (like dynamic configuration, full modularization, comprehensive unit tests for existing logic, and command-line arguments) were hindered by tool limitations in the automated development environment. Manual intervention would be needed to implement these.

## Features

*   Supports translation of DOCX, PDF, and plain text files (as per `read_file` function).
*   Uses a multi-agent pipeline (Translator, Improver, Quality Checker, Reviser) for translation.
*   Performs document summarization and category extraction.
*   Utilizes local LLMs via Ollama (or any OpenAI API-compatible endpoint).
*   Preserves document structure (titles, paragraphs) from DOCX and PDF where possible.
*   Basic test structure (`tests/test_example.py`) has been created.

## Setup

1.  **Prerequisites**:
    *   Python 3.8+
    *   An Ollama server (or other compatible LLM endpoint) running. Visit [https://ollama.com/](https://ollama.com/) for setup instructions.
    *   Ensure the desired LLM model (e.g., `deepseek-r1:32b-qwen-distill-q8_0`) is pulled and accessible via Ollama.

2.  **Clone the Repository**:
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```

3.  **Install Dependencies**:
    *   **Critical Missing Dependencies**: The current `requirements.txt` is missing several packages required by `trial_integrated4.py` (e.g., `pyautogen`, `ollama`, `python-docx`, `unstructured`, `retry`). You will need to install these manually:
        ```bash
        pip install pyautogen ollama python-docx unstructured retry tiktoken langchain-text-splitters tqdm PyYAML
        # Add any other missing ones based on errors at runtime.
        ```
    *   Then, install the remaining listed dependencies (though some might be redundant or for `app.py`):
        ```bash
        pip install -r requirements.txt
        ```
    *(Automated modification of `requirements.txt` failed due to tool limitations.)*

4.  **Configure `trial_integrated4.py` (Manual Steps Required)**:
    *   **NGROK Setup (Crucial for Local LLMs)**:
        *   If using a local Ollama server, you need to expose it to the internet (e.g., using ngrok).
        *   Install ngrok ([https://ngrok.com/download](https://ngrok.com/download)).
        *   Expose your local Ollama server (typically port 11434): `ngrok http 11434`
        *   Note the public URL ngrok provides (e.g., `https://your_unique_id.ngrok-free.app`).
    *   **Update Script Configuration**:
        *   Open `trial_integrated4.py`.
        *   The script's configuration (LLM endpoint URLs, model names, file paths) is **hardcoded**. Automated attempts to move these to a `config.yaml` and make the script read from it failed due to tool limitations.
        *   You **must manually update** these hardcoded values:
            *   `base_url` for the `OpenAI` class constructor (used by `ollama.Client`): Should be your ngrok URL (e.g., `https://your_unique_id.ngrok-free.app`).
            *   `base_url` within the `config_list` variable (used by Autogen agents): Should be your ngrok URL with `/v1` appended (e.g., `https://your_unique_id.ngrok-free.app/v1`).
            *   Default model name (e.g., `deepseek-r1:32b-qwen-distill-q8_0`), source/target languages, input/output file paths, etc., may also need direct modification in the script.

## Usage

Currently, the script runs with hardcoded input and output file paths due to limitations in implementing command-line arguments.

1.  **Verify Configuration**: Ensure you have manually updated the hardcoded ngrok URLs and other settings in `trial_integrated4.py`.
2.  **Place your input file**: Ensure the input file (e.g., `Surrender_No_preface.docx`, as hardcoded in `rel_path_to_text`) is in the root directory, or update this path in the script.
3.  **Run the script**:
    ```bash
    python trial_integrated4.py
    ```
4.  **Output**: The translated document will be saved as configured (e.g., `output_deepseek_32.docx`, as hardcoded).

## Code Structure Overview

*   `trial_integrated4.py`: Main script for translation, agent definitions, document processing.
*   `app.py`: Separate Streamlit application for CSV data analysis.
*   `requirements.txt`: Python dependencies (currently incomplete for `trial_integrated4.py`).
*   `.gitignore`: Specifies intentionally untracked files by Git.
*   `LICENSE`: Contains the MIT license for the project.
*   `tests/`: Directory containing placeholder unit tests.

## Limitations & Future Work

*   **Configuration**: Settings are hardcoded. External configuration (e.g., `config.yaml` or CLI args) is needed.
*   **Dependency Management**: `requirements.txt` needs to be corrected.
*   **Error Handling**: Could be more specific.
*   **Testing**: Meaningful unit tests for existing logic are missing.
*   **Modularity**: `trial_integrated4.py` is monolithic.
*   **CLI Arguments**: Not implemented.

Contributions addressing these are welcome.

## License

MIT License (see `LICENSE` file).
