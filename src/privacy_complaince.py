import os
import json
import logging
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from prompt import load_privacy_prompt_template
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def load_conversations(directory_path: str) -> List[Dict[str, Any]]:
    """
    Load all JSON conversation files from the specified directory.

    Args:
        directory_path (str): Directory path containing JSON conversation files.

    Returns:
        List[Dict]: List of conversations with 'call_id' and 'conversation' keys.
    """
    conversations = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
                    conversations.append({
                        'call_id': os.path.splitext(filename)[0],
                        'conversation': conversation
                    })
                    logging.info(f"Loaded conversation file: {filename}")
            except Exception as e:
                logging.error(f"Failed to load {filename}: {e}")
    return conversations

def load_single_conversation(filepath: str) -> Dict[str, Any]:
    """
    Load a single conversation JSON file.

    Args:
        filepath (str): Path to the conversation JSON file.

    Returns:
        Dict: Dictionary with 'call_id' and 'conversation' keys.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
            call_id = os.path.splitext(os.path.basename(filepath))[0]
            logging.info(f"Loaded single conversation file: {filepath}")
            return {'call_id': call_id, 'conversation': conversation}
    except Exception as e:
        logging.error(f"Failed to load single conversation file {filepath}: {e}")
        raise

def format_conversation_for_prompt(conversation: List[Dict[str, Any]]) -> str:
    """
    Format a single conversation into text for LLM prompt.

    Args:
        conversation (List[Dict]): List of utterances with 'speaker' and 'text'.

    Returns:
        str: Formatted conversation text.
    """
    lines = []
    for utterance in conversation:
        speaker = utterance.get('speaker', 'unknown').capitalize()
        text = utterance.get('text', '').strip()
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)

def build_user_prompt(batch: List[Dict[str, Any]]) -> str:
    """
    Build a user prompt for the LLM with batch conversations.

    Args:
        batch (List[Dict]): List of conversations in the batch.

    Returns:
        str: User prompt string.
    """
    prompt = "Analyze the following conversations for privacy and compliance violations as per the system instructions.\n\n"
    for conv in batch:
        prompt += f"Call ID: {conv['call_id']}\n"
        prompt += format_conversation_for_prompt(conv['conversation']) + "\n\n"
    return prompt

def analyze_batch(llm_client: ChatGroq, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze a batch of conversations by invoking the LLM.

    Args:
        llm_client: An initialized LangChain Groq LLM client.
        batch (List[Dict]): List of batch conversations.

    Returns:
        List[Dict]: Parsed LLM analysis results per conversation.
    """
    user_prompt = build_user_prompt(batch)
    messages = [
        ("system", load_privacy_prompt_template()),
        ("human", user_prompt)
    ]
    logging.info(f"Sending batch of {len(batch)} conversations to LLM...")
    response = llm_client.invoke(messages)
    try:
        analysis = json.loads(response.content)
    except Exception as e:
        logging.error(f"Failed to parse LLM response: {e}")
        analysis = [{
            "call_id": conv['call_id'],
            "violation": "error",
            "explanation": "Failed to parse LLM output",
            "terms": ""
        } for conv in batch]
    return analysis

def analyze_single(llm_client: ChatGroq, conversation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single conversation by invoking the LLM.

    Args:
        llm_client: An initialized LangChain Groq LLM client.
        conversation (Dict): A conversation dictionary with 'call_id' and 'conversation'.

    Returns:
        Dict: Parsed LLM analysis result for the single conversation.
    """
    batch = [conversation]
    results = analyze_batch(llm_client, batch)
    return results[0] if results else {}

def save_results_to_csv(results: List[Dict[str, Any]], filepath: str):
    """
    Save analysis results to a CSV file.

    Args:
        results (List[Dict]): List of analysis results.
        filepath (str): File path to save the CSV report.
    """
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    logging.info(f"Results saved to CSV file: {filepath}")

def main():
    """
    Main function to execute privacy and compliance detection either on a batch of conversations
    or a single conversation file.

    Hardcoded paths are used for simplicity:
    - Batch mode: processes all JSON conversation files in specified directory in batches of 25
    - Single mode: processes a single JSON file

    Configure the API key by setting the environment variable 'GROQ_API_KEY4' before running.

    Results are saved as CSV files in the './output/' directory.
    """

    # Setup paths and parameters
    BATCH_DIR = "./data/All_Conversations"
    SINGLE_FILE_PATH = "./data/All_Conversations/acec9db1-1209-4f98-881e-6d413e1aba2d.json"
    OUTPUT_DIR = "./output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    BATCH_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "privacy_compliance_batch_report.csv")
    SINGLE_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "privacy_compliance_single_report.csv")
    BATCH_SIZE = 25
    API_KEY = os.getenv("GROQ_API_KEY4")

    if not API_KEY:
        logging.error("Environment variable 'GROQ_API_KEY4' not set. Exiting.")
        sys.exit(1)

    llm_client = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=API_KEY)

    # Switch between modes by setting this flag
    PROCESS_SINGLE_FILE = True  # Set True to process the single file instead of batch

    if PROCESS_SINGLE_FILE:
        logging.info(f"Processing single conversation file: {SINGLE_FILE_PATH}")
        try:
            conversation = load_single_conversation(SINGLE_FILE_PATH)
            result = analyze_single(llm_client, conversation)
            save_results_to_csv([result], SINGLE_OUTPUT_CSV)
        except Exception as e:
            logging.error(f"Failed processing single file: {e}")
    else:
        logging.info(f"Processing all conversations in directory: {BATCH_DIR}")
        conversations = load_conversations(BATCH_DIR)
        all_results = []
        for i in tqdm(range(0, len(conversations), BATCH_SIZE), desc="Processing batches"):
            batch = conversations[i:i + BATCH_SIZE]
            try:
                batch_results = analyze_batch(llm_client, batch)
                all_results.extend(batch_results)
            except Exception as e:
                logging.error(f"Error processing batch starting at index {i}: {e}")
                # Continue processing remaining batches
        save_results_to_csv(all_results, BATCH_OUTPUT_CSV)

if __name__ == "__main__":
    main()
