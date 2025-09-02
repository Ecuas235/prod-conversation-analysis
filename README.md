# Debt Collection Call Analysis

This project delivers a professional Streamlit application to analyze debt collection call conversations with a focus on **compliance**, **professionalism**, and **call quality metrics**. The app integrates advanced NLP and regex-based techniques to detect profanity and privacy violations, complemented by insightful call quality visualizations.

## Features

- **Profanity Detection**  
  Hybrid approach combining regex pattern matching with large language model (LLM) prompting for precise detection of offensive language by agents and borrowers.

- **Privacy & Compliance Violation Detection**  
  Leveraging a fine-tuned LLM prompt system to identify instances where sensitive information is disclosed without prior borrower identity verification.

- **Call Quality Metrics Analysis**  
  Calculates overtalk (simultaneous speaking) and silence percentages using utterance timestamps, with multiple interactive visualizations illustrating call dynamics.

## Installation

To install the required Python packages for this project, run:

```bash
pip install -r requirements.txt
```

## Usage

The app accepts **one YAML or JSON conversation file at a time**, with detailed utterances including speaker labels and timestamps.

Run the application locally with:

```bash
streamlit run src/debt_app.py
```

Use the sidebar to select the analysis entity:

- Profanity Detection
- Privacy & Compliance Violation
- Call Quality Metrics

Upload your conversation file and receive instant, comprehensive results displayed in clean tables and interactive charts.

## Implementation Notes

- The profanity detection employs a dual regex + LLM method to balance speed and subtlety in detection.
- Privacy compliance relies on secure access to an LLM API, configurable via the environment variable 
- Call quality metrics utilize precise time-based computations with visual aids leveraging Plotly for clear interpretation.

## Deliverables

- Modular, well-documented Python code built for maintainability and extensibility.
- A polished Streamlit interface catering to both technical and non-technical users.
- An efficient pipeline bridging classical NLP and state-of-the-art LLM capabilities.
- Visual reports and charts that make deep call analytics accessible at a glance.
