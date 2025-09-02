import re
import os
import json
import logging
from typing import Dict, List, Set, Any, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tqdm import tqdm
from prompt import load_profanity_prompt_template

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()  # Load environment variables from .env file


def load_conversation(filepath: str) -> Dict[str, Any]:
    """
    Load a single conversation JSON file.

    Args:
        filepath (str): Path to the JSON file containing a single conversation.

    Returns:
        Dict: A dictionary with 'call_id' and 'conversation' (list of utterance dicts).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
        call_id = filepath.split(os.sep)[-1].split('.')[0]
        logging.info(f"Loaded conversation from {filepath} with call_id: {call_id}")
        return {'call_id': call_id, 'conversation': conversation}
    except Exception as e:
        logging.error(f"Error loading conversation file {filepath}: {e}")
        raise


def load_profanity_patterns(profanity_file: str) -> List:
    """
    Load profanity words from a file and compile regex patterns.

    Args:
        profanity_file (str): Path to a file containing profanity words, one per line.

    Returns:
        List of tuples: (word, compiled_regex)
    """
    try:
        with open(profanity_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        patterns = [(word, re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)) for word in words]
        logging.info(f"Loaded {len(patterns)} profanity patterns from {profanity_file}")
        return patterns
    except Exception as e:
        logging.error(f"Error loading profanity patterns from {profanity_file}: {e}")
        raise


def detect_profanity_regex(text: str, profanity_patterns: List) -> Set[str]:
    """
    Detect profane words in text using regex patterns.

    Args:
        text (str): Input text to check.
        profanity_patterns (List): List of (word, compiled_regex) tuples.

    Returns:
        Set[str]: Set of profane words detected.
    """
    detected = set()
    for word, pattern in profanity_patterns:
        if pattern.search(text):
            detected.add(word)
    if detected:
        logging.debug(f"Regex detected profanities: {detected} in text: {text}")
    return detected


def detect_profanity_with_llm(text: str, llm_client: ChatGroq, prompt_template: str) -> Set[str]:
    """
    Use an LLM to detect subtle or contextual profanity in given text using a prompt template.
    Uses .invoke method with system prompt and user prompt as conversation.

    Args:
        text (str): Text to analyze.
        llm_client (ChatGroq): Initialized LLM client.
        prompt_template (str): The prompt template for profanity detection.

    Returns:
        Set[str]: Set of detected profane words if any, otherwise empty set.
    """
    system_prompt = prompt_template
    user_prompt = f'Analyze this text for any profanity or offensive language: "{text}"\n' \
                  f'If profanity or offensive language is found, return only a valid JSON object with keys ' \
                  f'"detected" and "terms" (list of exact profane words). If none found, return ' \
                  f'{{"detected": false, "terms": []}}.'

    try:
        response = llm_client.invoke([("system", system_prompt), ("user", user_prompt)])
        content = response.content
        result = json.loads(content)
        if result.get("detected", False):
            profane_terms = {term.lower() for term in result.get("terms", [])}
            logging.debug(f"LLM detected profanities: {profane_terms} in text: {text}")
            return profane_terms
    except Exception as e:
        logging.warning(f"LLM profanity check failed: {e}")
    return set()


def process_conversation(
    conversation_data: Dict[str, Any],
    profanity_patterns: List,
    use_llm: bool = False,
    llm_api_key: Optional[str] = None,
    prompt_template: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process one conversation to detect profanity in each utterance,
    combining outputs from regex and LLM (if enabled).
    Profane terms from both methods are merged per utterance.

    Args:
        conversation_data (Dict): Dict with 'call_id' and 'conversation' (list of utterances).
        profanity_patterns (List): List of (word, compiled_regex) tuples.
        use_llm (bool): Whether to use LLM fallback detection (default False).
        llm_api_key (str, optional): API key for LLM. Required if use_llm is True.
        prompt_template (str, optional): Prompt template string for LLM.

    Returns:
        Dict: profanity detection results for the conversation.
    """
    llm_client = None
    if use_llm:
        if not llm_api_key:
            raise ValueError("LLM API key required when use_llm=True")
        llm_client = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=llm_api_key)
        if prompt_template is None:
            prompt_template = load_profanity_prompt_template()
        logging.info("LLM enabled for profanity detection.")

    results_utterances = []
    agent_profane_words = set()
    customer_profane_words = set()

    utterances = conversation_data.get("conversation", [])
    call_id = conversation_data.get("call_id", "unknown")
    logging.info(f"Processing conversation with call_id: {call_id}")

    for idx, utt in enumerate(utterances, start=1):
        text = utt.get("text", "")
        speaker = utt.get("speaker", "").lower()

        profane_words_regex = detect_profanity_regex(text, profanity_patterns)

        profane_words_llm = set()
        if use_llm and llm_client:
            profane_words_llm = detect_profanity_with_llm(text, llm_client, prompt_template)

        combined_profane_words = profane_words_regex.union(profane_words_llm)

        has_profanity = len(combined_profane_words) > 0
        if has_profanity:
            if speaker == "agent":
                agent_profane_words.update(combined_profane_words)
            elif speaker in {"customer", "borrower"}:
                customer_profane_words.update(combined_profane_words)

        results_utterances.append({
            "utterance_number": idx,
            "speaker": speaker,
            "text": text,
            "has_profanity": has_profanity,
            "profane_words": sorted(combined_profane_words)
        })

    summary = {
        "total_utterances": len(utterances),
        "calls_with_profanity": len(agent_profane_words) > 0 or len(customer_profane_words) > 0,
        "agent_profane_words": sorted(agent_profane_words),
        "customer_profane_words": sorted(customer_profane_words)
    }

    logging.info(f"Completed processing conversation {call_id}.")
    return {
        "call_id": call_id,
        "utterances": results_utterances,
        "summary": summary
    }


def save_profanity_report_csv(output: Dict[str, Any], csv_filepath: str) -> None:
    """
    Save profanity detection output as a detailed CSV report.

    Args:
        output (Dict): Profanity detection output.
        csv_filepath (str): Destination CSV file path.
    """
    records = []
    call_id = output.get("call_id", "unknown")
    for utterance in output.get("utterances", []):
        records.append({
            "call_id": call_id,
            "utterance_number": utterance["utterance_number"],
            "speaker": utterance["speaker"],
            "text": utterance["text"],
            "has_profanity": utterance["has_profanity"],
            "profane_words": ", ".join(utterance["profane_words"]) if utterance["profane_words"] else ""
        })
    df = pd.DataFrame(records)
    df.to_csv(csv_filepath, index=False)
    logging.info(f"CSV profanity report saved to {csv_filepath}")


def load_all_conversations(directory_path: str) -> List[Dict[str, Any]]:
    """
    Load all conversations (JSON files) from a specified directory.

    Args:
        directory_path (str): Path to directory with JSON conversation files.

    Returns:
        List of dicts: Each dict contains 'call_id' and 'conversation' keys.
    """
    conversations = []
    files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
    logging.info(f"Found {len(files)} JSON files in directory {directory_path} to process.")
    for f in files:
        filepath = os.path.join(directory_path, f)
        try:
            conv = load_conversation(filepath)
            conversations.append(conv)
        except Exception as e:
            logging.warning(f"Skipping file {filepath} due to error: {e}")
    return conversations


def detect_profanity_llm_batch(
    batch_conversations: List[Dict[str, Any]],
    llm_client: ChatGroq,
    prompt_template: str
) -> List[Dict[str, Any]]:
    """
    Batch detect profanity in multiple conversations using a single LLM invocation.

    Args:
        batch_conversations (List[Dict]): List of conversation dicts with 'call_id' and 'conversation'.
        llm_client (ChatGroq): Initialized LLM client.
        prompt_template (str): The profanity detection prompt.

    Returns:
        List[Dict]: LLM profanity detection results per conversation.
    """
    # Construct prompt for batch
    system_msg = prompt_template
    user_msg = "# Batch profanity detection\n"
    for idx, conv in enumerate(batch_conversations, start=1):
        user_msg += f"\nConversation {idx} ID: {conv['call_id']}\n"
        for i, utt in enumerate(conv['conversation'], start=1):
            speaker = utt.get('speaker', 'unknown').capitalize()
            text = utt.get('text', '').replace('\n', ' ')
            user_msg += f"{i}. [{speaker}] {text}\n"

    user_msg += "\nPlease respond with a JSON array of objects:\n" \
                "[{\"call_id\": \"string\", \"utterances\": [{\"utterance_number\": int, \"has_profanity\": bool, \"profane_words\": [\"word1\"]}]}, ...]\n" \
                "Only respond with valid JSON array."

    response = llm_client.invoke([("system", system_msg), ("user", user_msg)])
    try:
        results = json.loads(response.content)
    except Exception as e:
        logging.warning(f"Failed to parse LLM batch response: {e}")
        results = []
    return results


def process_conversations_in_batches(
        conversations: List[Dict[str, Any]],
        profanity_patterns: List,
        batch_size: int = 25,
        use_llm: bool = False,
        llm_api_key: Optional[str] = None,
        prompt_template: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process all conversations in batches, batching LLM calls wherever LLM is used.
    Regex detection applied individually before LLM batch for efficiency.

    Args:
        conversations (List[Dict]): List of conversation dicts with 'call_id' and 'conversation'.
        profanity_patterns (List): List of (word, compiled_regex) tuples.
        batch_size (int): Number of conversations to process per batch.
        use_llm (bool): Whether to use LLM batch detection.
        llm_api_key (Optional[str]): API key for LLM.
        prompt_template (Optional[str]): Prompt template string for LLM.

    Returns:
        List[Dict]: Combined profanity detection outputs per conversation.
    """
    llm_client = None
    if use_llm:
        if not llm_api_key:
            raise ValueError("LLM API key required when use_llm=True")
        llm_client = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=llm_api_key)
        if prompt_template is None:
            prompt_template = load_profanity_prompt_template()
        logging.info("LLM enabled for batch profanity detection.")

    all_results = []
    total = len(conversations)
    for i in range(0, total, batch_size):
        batch = conversations[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1} / {(total // batch_size) + 1} with {len(batch)} conversations")

        # First do regex-based detection per conversation / utterance
        regex_only_results = []
        for conv in batch:
            call_id = conv['call_id']
            utterances = conv['conversation']
            utter_results = []
            for idx, utt in enumerate(utterances, start=1):
                text = utt.get('text', '')
                speaker = utt.get('speaker', '').lower()
                profane_words_regex = detect_profanity_regex(text, profanity_patterns)
                utter_results.append({
                    "utterance_number": idx,
                    "speaker": speaker,
                    "text": text,
                    "profane_words_regex": profane_words_regex,
                })
            regex_only_results.append({"call_id": call_id, "utterances": utter_results})

        # If LLM is not used, finalize results here
        if not use_llm:
            for r in regex_only_results:
                call_id = r["call_id"]
                utterances_data = r["utterances"]
                agent_profanity = set()
                customer_profanity = set()
                utterances_out = []
                for utt in utterances_data:
                    combined_profanity = {w.lower() for w in utt["profane_words_regex"]}
                    has_profanity = len(combined_profanity) > 0
                    if has_profanity:
                        if utt["speaker"] == "agent":
                            agent_profanity.update(combined_profanity)
                        elif utt["speaker"] in {"customer", "borrower"}:
                            customer_profanity.update(combined_profanity)
                    utterances_out.append({
                        "utterance_number": utt["utterance_number"],
                        "speaker": utt["speaker"],
                        "text": utt["text"],
                        "has_profanity": has_profanity,
                        "profane_words": sorted(combined_profanity),
                    })

                summary = {
                    "total_utterances": len(utterances_out),
                    "calls_with_profanity": len(agent_profanity) > 0 or len(customer_profanity) > 0,
                    "agent_profane_words": sorted(agent_profanity),
                    "customer_profane_words": sorted(customer_profanity)
                }

                all_results.append({"call_id": call_id, "utterances": utterances_out, "summary": summary})

            continue

        # Else, batch call LLM with the full batch text
        llm_outputs = detect_profanity_llm_batch(batch, llm_client, prompt_template)

        llm_map = {entry["call_id"]: entry for entry in llm_outputs}

        # Combine regex and LLM results per utterance
        for regex_conv in regex_only_results:
            call_id = regex_conv["call_id"]
            llm_conv = llm_map.get(call_id, {"utterances": []})
            agent_profanity = set()
            customer_profanity = set()

            utterances_out = []
            for regex_utt, llm_utt in zip(regex_conv["utterances"], llm_conv.get("utterances", [])):
                regex_terms = {w.lower() for w in regex_utt.get("profane_words_regex", set())}
                llm_terms = {w.lower() for w in llm_utt.get("profane_words", [])}
                combined_terms = regex_terms.union(llm_terms)
                has_profanity = len(combined_terms) > 0

                speaker = regex_utt["speaker"]
                if has_profanity:
                    if speaker == "agent":
                        agent_profanity.update(combined_terms)
                    elif speaker in {"customer", "borrower"}:
                        customer_profanity.update(combined_terms)

                utterances_out.append({
                    "utterance_number": regex_utt["utterance_number"],
                    "speaker": speaker,
                    "text": regex_utt["text"],
                    "has_profanity": has_profanity,
                    "profane_words": sorted(combined_terms)
                })

            summary = {
                "total_utterances": len(utterances_out),
                "calls_with_profanity": len(agent_profanity) > 0 or len(customer_profanity) > 0,
                "agent_profane_words": sorted(agent_profanity),
                "customer_profane_words": sorted(customer_profanity)
            }
            all_results.append({"call_id": call_id, "utterances": utterances_out, "summary": summary})

    return all_results


def all_conversations_usage():
        conversations = load_all_conversations(CONVERSATIONS_DIR)
        prompt_template = load_profanity_prompt_template()
        all_results = process_conversations_in_batches(
            conversations,
            profanity_patterns,
            batch_size=25,
            use_llm=USE_LLM,
            llm_api_key=LLM_API_KEY,
            prompt_template=prompt_template
        )

        # Aggregate all utterances into one DataFrame
        records = []
        for res in all_results:
            call_id = res.get("call_id", "unknown")
            for utt in res.get("utterances", []):
                records.append({
                    "call_id": call_id,
                    "utterance_number": utt["utterance_number"],
                    "speaker": utt["speaker"],
                    "text": utt["text"],
                    "has_profanity": utt["has_profanity"],
                    "profane_words": ", ".join(utt["profane_words"]) if utt["profane_words"] else ""
                })
        df_all = pd.DataFrame(records)
        combined_csv_path = os.path.join(output_dir, "all_conversations_profanity_report.csv")
        df_all.to_csv(combined_csv_path, index=False)
        logging.info(f"All conversations combined report saved to {combined_csv_path}")

def single_file_usage(filepath: str):
    conversation = load_conversation(filepath)
    prompt_template = load_profanity_prompt_template()
    output = process_conversation(
        conversation,
        profanity_patterns,
        use_llm=USE_LLM,
        llm_api_key=LLM_API_KEY,
        prompt_template=prompt_template
    )
    call_id = output.get("call_id", "unknown")
    csv_path = os.path.join(output_dir, f"{call_id}_profanity_report.csv")
    save_profanity_report_csv(output, csv_path)
    logging.info(f"Single conversation report saved to {csv_path}")
    
    
if __name__ == "__main__":
    CONVERSATIONS_DIR = "./data/All_Conversations"
    PROFANITY_WORDLIST = "./data/bad-words.txt"
    USE_LLM = False
    LLM_API_KEY = os.getenv("GROQ_API_KEY4")
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    profanity_patterns = load_profanity_patterns(PROFANITY_WORDLIST)
    # all_conversations_usage()

    # Example single file usage
    example_file = os.path.join(CONVERSATIONS_DIR, "b70866a2-2f46-4784-992b-74d6dc60806e.json")
    single_file_usage(example_file)

