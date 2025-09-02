import streamlit as st
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # For additional visualizations

import profanity  # profanity.py module
import privacy_complaince # privacy_complaince.py module
from prompt import load_profanity_prompt_template

st.set_page_config(
    page_title="Debt Collection Call Analysis",
    layout="centered",
    initial_sidebar_state="expanded",
)


def calculate_overtalk_and_silence(
    utterances: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Calculate overtalk % and silence % given utterances with stime and etime.

    Overtalk: total time overlapping by multiple speakers
    Silence: gaps between utterances with no speech

    Returns percentages relative to total call duration.
    """

    intervals = []
    for utt in utterances:
        stime = utt.get("stime")
        etime = utt.get("etime")
        speaker = utt.get("speaker")
        if stime is None or etime is None:
            continue
        intervals.append({"start": stime, "end": etime, "speaker": speaker})

    if not intervals:
        return 0.0, 0.0

    intervals.sort(key=lambda x: x["start"])

    call_start = intervals[0]["start"]
    call_end = max(i["end"] for i in intervals)
    call_duration = call_end - call_start
    if call_duration <= 0:
        return 0.0, 0.0

    timeline = []
    for iv in intervals:
        timeline.append((iv["start"], "start", iv["speaker"]))
        timeline.append((iv["end"], "end", iv["speaker"]))

    timeline.sort()

    active_speakers = set()
    last_time = None
    overtalk_time = 0.0

    for time, typ, speaker in timeline:
        if last_time is not None and time > last_time:
            if len(active_speakers) > 1:
                overtalk_time += time - last_time
        if typ == "start":
            active_speakers.add(speaker)
        else:
            active_speakers.discard(speaker)
        last_time = time

    merged = []
    for iv in intervals:
        merged.append((iv["start"], iv["end"]))

    merged.sort()
    merged_intervals = []
    cur_start, cur_end = merged[0]
    for s, e in merged[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged_intervals.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged_intervals.append((cur_start, cur_end))

    voiced_time = sum(e - s for s, e in merged_intervals)
    silence_time = call_duration - voiced_time

    overtalk_pct = (overtalk_time / call_duration) * 100
    silence_pct = (silence_time / call_duration) * 100

    return overtalk_pct, silence_pct


st.title("Debt Collection Conversation Analysis")
st.markdown(
    """
    This application analyzes *one conversation file at a time* for:
    - **Profanity Detection** (Hybrid Regex + LLM)
    - **Privacy & Compliance Violation** (LLM)
    - **Call Quality Metrics** (Overtalk and Silence % with visualization)
    
    Upload a valid conversation JSON file containing utterances with timestamps.
    """
)

entity_option = st.sidebar.selectbox(
    "Select Entity to Analyze",
    options=[
        "Profanity Detection",
        "Privacy and Compliance Violation",
        "Call Quality Metrics",
    ],
)

approach_map = {
    "Profanity Detection": "Hybrid Regex + LLM",
    "Privacy and Compliance Violation": "LLM Prompt System",
    "Call Quality Metrics": "Timestamp-Based Calculation",
}
approach_option = approach_map.get(entity_option, "")

# st.sidebar.markdown(f"Selected Approach: **{approach_option}** (fixed based on task requirements)")

uploaded_file = st.file_uploader(
    "Upload a single conversation JSON file", type=["json", "yaml", "yml"]
)


def analyze_profanity(file_path: str) -> Dict[str, Any]:
    conversation = profanity.load_conversation(file_path)
    profanity_patterns = profanity.load_profanity_patterns("./data/bad-words.txt")
    llm_key = os.getenv("GROQ_API_KEY4")
    if not llm_key:
        st.error("Environment variable 'GROQ_API_KEY4' not set. LLM approach cannot run.")
        return {}
    prompt_template = load_profanity_prompt_template()
    result = profanity.process_conversation(
        conversation,
        profanity_patterns,
        use_llm=True,
        llm_api_key=llm_key,
        prompt_template=prompt_template,
    )
    return result


def analyze_privacy(file_path: str) -> Dict[str, Any]:
    conversation = privacy_complaince.load_single_conversation(file_path)
    api_key = os.getenv("GROQ_API_KEY4")
    if not api_key:
        st.error("Environment variable 'GROQ_API_KEY4' not set. Cannot perform LLM-based privacy analysis.")
        return {}
    llm_client = privacy_complaince.ChatGroq(
        model="openai/gpt-oss-20b", temperature=0, api_key=api_key
    )
    result = privacy_complaince.analyze_single(llm_client, conversation)
    return result


def display_profanity_result(result: Dict[str, Any]):
    if not result:
        st.info("No results to display.")
        return

    summary = result.get("summary", {})
    st.subheader("Profanity Detection Summary")
    st.write(f"- Total Utterances: {summary.get('total_utterances', 0)}")
    st.write(f"- Call Contains Profanity: {summary.get('calls_with_profanity', False)}")
    st.write(f"- Agent Profane Words: {', '.join(summary.get('agent_profane_words', []))}")
    st.write(f"- Borrower Profane Words: {', '.join(summary.get('customer_profane_words', []))}")

    st.subheader("Utterances with Profanity Details")
    utterances = result.get("utterances", [])
    if utterances:
        records = []
        for utt in utterances:
            records.append(
                {
                    "Utterance #": utt["utterance_number"],
                    "Speaker": utt["speaker"].capitalize(),
                    "Has Profanity": utt["has_profanity"],
                    "Profane Words": ", ".join(utt["profane_words"]) if utt["profane_words"] else "",
                }
            )
        df = pd.DataFrame(records)
        st.dataframe(df)
    else:
        st.info("No profanities detected in utterances.")


def display_privacy_result(result: Dict[str, Any]):
    if not result:
        st.info("No results to display.")
        return
    st.subheader("Privacy and Compliance Violation Result")
    st.write(f"- Call ID: {result.get('call_id', 'unknown')}")
    st.write(f"- Privacy Violation: {result.get('violation', 'Unknown')}")
    st.write(f"- Explanation: {result.get('explanation', 'No explanation available')}")
    st.write(f"- Key Terms: {result.get('terms', '')}")


def display_call_quality_result(result: Dict[str, Any], filepath: str):
    import json

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            conversation_data = json.load(f)
        except Exception as e:
            st.error(f"Failed to load conversation JSON: {e}")
            return

    if isinstance(conversation_data, dict) and "conversation" in conversation_data:
        utterances = conversation_data["conversation"]
    else:
        utterances = conversation_data

    overtalk_pct, silence_pct = calculate_overtalk_and_silence(utterances)
    total_utterances = len(utterances)

    st.subheader("Call Quality Metrics Analysis")
    st.write(f"Total utterances in call: {total_utterances}")
    st.write(f"Overtalk percentage: **{overtalk_pct:.2f}%**")
    st.write(f"Silence percentage: **{silence_pct:.2f}%**")

    fig_pie = go.Figure(
        go.Pie(
            labels=["Overtalk", "Silence", "Normal Speech"],
            values=[overtalk_pct, silence_pct, 100 - overtalk_pct - silence_pct],
            marker=dict(colors=["#FF4136", "#0074D9", "#2ECC40"]),
            hoverinfo="label+percent",
            textinfo="label+percent",
        )
    )
    fig_pie.update_layout(
        title_text="Call Speech Segments Distribution",
        margin=dict(t=40, b=0, l=0, r=0),
        height=400,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Additional Visualizations to Impress

    # 1. Utterance Duration Distribution Histogram
    durations = []
    speakers = []
    for utt in utterances:
        stime = utt.get("stime")
        etime = utt.get("etime")
        if stime is not None and etime is not None:
            durations.append(etime - stime)
            speakers.append(utt.get("speaker", "unknown").capitalize())
    df_durations = pd.DataFrame({"Duration": durations, "Speaker": speakers})

    fig_hist = px.histogram(
        df_durations,
        x="Duration",
        color="Speaker",
        barmode="overlay",
        nbins=20,
        title="Utterance Duration Distribution by Speaker",
        labels={"Duration": "Utterance Duration (seconds)"},
        opacity=0.75,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # 2. Speaking Time by Speaker Bar Chart
    df_durations_sum = (
        df_durations.groupby("Speaker")["Duration"].sum().reset_index()
    )
    fig_bar = px.bar(
        df_durations_sum,
        x="Speaker",
        y="Duration",
        title="Total Speaking Time by Speaker",
        labels={"Duration": "Total Speaking Time (seconds)"},
        color="Speaker",
        color_discrete_map={"Agent": "#0074D9", "Customer": "#2ECC40", "Unknown": "#AAAAAA"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Overtalk Timeline Chart (optional detailed timeline)
    # Construct timeline data showing simultaneous speaking time by second slice
    # For impressiveness, show timeline as stacked area chart by speaker + overtalk overlay

    # Prepare timeline in 1-second resolution
    timeline_start = min(d["start"] for d in [{"start": utt.get("stime")} for utt in utterances if utt.get("stime") is not None])
    timeline_end = max(d["end"] for d in [{"end": utt.get("etime")} for utt in utterances if utt.get("etime") is not None])
    total_seconds = int(timeline_end - timeline_start) + 1

    time_points = list(range(total_seconds))
    agent_speaking = [0] * total_seconds
    customer_speaking = [0] * total_seconds
    overlap = [0] * total_seconds

    for utt in utterances:
        stime = utt.get("stime")
        etime = utt.get("etime")
        speaker = utt.get("speaker", "").lower()
        if stime is None or etime is None:
            continue
        start_idx = int(stime - timeline_start)
        end_idx = int(etime - timeline_start)
        for t in range(start_idx, min(end_idx + 1, total_seconds)):
            if speaker == "agent":
                agent_speaking[t] = 1
            elif speaker in ("customer", "borrower"):
                customer_speaking[t] = 1

    for i in range(total_seconds):
        if agent_speaking[i] == 1 and customer_speaking[i] == 1:
            overlap[i] = 1

    normal_speech = [
        1 if (agent_speaking[i] == 1 or customer_speaking[i] == 1) and overlap[i] == 0 else 0
        for i in range(total_seconds)
    ]

    df_timeline = pd.DataFrame({
        "Time (s)": time_points,
        "Agent Speaking": agent_speaking,
        "Customer Speaking": customer_speaking,
        "Overtalk": overlap,
        "Normal Speech": normal_speech,
    })

    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=df_timeline["Time (s)"],
        y=df_timeline["Agent Speaking"],
        mode='lines',
        name='Agent Speaking',
        stackgroup='one',
        line=dict(color="#0074D9"),
    ))
    fig_area.add_trace(go.Scatter(
        x=df_timeline["Time (s)"],
        y=df_timeline["Customer Speaking"],
        mode='lines',
        name='Customer Speaking',
        stackgroup='one',
        line=dict(color="#2ECC40"),
    ))
    fig_area.add_trace(go.Scatter(
        x=df_timeline["Time (s)"],
        y=df_timeline["Overtalk"],
        mode='lines',
        name='Overtalk',
        stackgroup='one',
        line=dict(color="#FF4136"),
        fill='tonexty',
    ))

    fig_area.update_layout(
        title="Speaking and Overtalk Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Speaking Indicator (Stacked)",
        height=400,
        legend_title="Segments",
        showlegend=True,
        yaxis=dict(range=[0,3])
    )
    st.plotly_chart(fig_area, use_container_width=True)


if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_path = f"./{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    st.markdown(f"### Analyzing file: {uploaded_file.name}")

    if entity_option == "Profanity Detection":
        with st.spinner("Performing hybrid profanity detection (Regex + LLM)..."):
            result = analyze_profanity(file_path)
            display_profanity_result(result)

    elif entity_option == "Privacy and Compliance Violation":
        with st.spinner("Performing LLM-based privacy compliance analysis..."):
            result = analyze_privacy(file_path)
            display_privacy_result(result)

    else:  # Call Quality Metrics
        with st.spinner("Calculating call quality metrics and producing visualizations..."):
            display_call_quality_result({}, file_path)

    try:
        os.remove(file_path)
    except Exception:
        pass

else:
    st.info("Please upload a conversation JSON file to start analysis.")

st.markdown("---")
st.markdown(
    """
**Instructions:**

- Upload one conversation file at a time in JSON format, containing the utterances with timestamps (`stime`, `etime`).

- Profanity detection uses a hybrid approach combining regex and LLM for best detection.

- Privacy violation detection is LLM-based, checking for disclosure of sensitive information without verification.

- Call Quality Metrics analyzes overtalk and silence percentages from utterance timestamps and visualizes them with multiple charts for deeper insights.

- Ensure environment variable `GROQ_API_KEY4` is set for LLM-based methods.
"""
)
