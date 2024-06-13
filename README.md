# LLM
# Natural Language Query Agent

This project demonstrates a Natural Language Query Agent capable of answering questions over a set of lecture notes and model architectures.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Instructions

1. Run the script:
    ```bash
    python query_agent.py
    ```

2. Example query:
    ```python
    query = "What are some milestone model architectures and papers in the last few years?"
    response = generate_response(query)
    print(response)
    ```

## Future Work

- Implement conversational memory for handling follow-up queries.
- Expand the dataset to include more lectures and model architectures.
- Utilize LLMs/Multi-modal models by means of API calls.
