import hashlib
import os

import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from pdf_utils import extract_chunks


# Configure the Streamlit page before displaying any UI elements.
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
)


def get_setting(name, default):
    """Read a setting from environment variables or Streamlit secrets."""
    environment_value = os.getenv(name)
    if environment_value and environment_value.strip():
        return environment_value.strip()

    try:
        secret_value = st.secrets.get(name, default)
    except Exception:
        # Local runs usually do not have a Streamlit secrets file.
        secret_value = default

    if secret_value is None:
        return default

    value = str(secret_value).strip()
    return value if value else default


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MODEL = get_setting("LLM_MODEL", DEFAULT_LLM_MODEL)
LLM_CPU_DTYPE = get_setting("LLM_CPU_DTYPE", "float32").lower()

try:
    MAX_PDF_SIZE_MB = max(1, int(get_setting("MAX_PDF_SIZE_MB", "20")))
except (TypeError, ValueError):
    MAX_PDF_SIZE_MB = 20

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
TOP_K = 3
SUMMARY_GROUP_SIZE = 6


st.title("Research Paper Assistant")
st.write(
    "Upload a research paper to get a simple summary "
    "or ask questions about its contents."
)


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedding_model():
    """Load and cache the embedding model on the available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device)


@st.cache_resource(show_spinner="Loading the language model...")
def load_llm(model_name, cpu_dtype_name):
    """Load and cache Qwen, using CUDA when available and CPU otherwise."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        dtype = torch.float16
        device = "cuda"
    else:
        cpu_dtypes = {
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = cpu_dtypes.get(cpu_dtype_name.lower(), torch.float32)
        device = "cpu"

    # Avoid automatic device mapping; explicitly place the fully loaded model.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=False,
    )
    model = model.to(device)
    model.eval()

    return tokenizer, model


def generate_text(
    tokenizer,
    model,
    system_prompt,
    user_prompt,
    max_new_tokens=250,
):
    """Generate text with the locally loaded language model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


def summarize_paper(chunks, tokenizer, model, progress_bar):
    """Summarize groups of PDF chunks and combine their notes."""
    groups = [
        chunks[index : index + SUMMARY_GROUP_SIZE]
        for index in range(0, len(chunks), SUMMARY_GROUP_SIZE)
    ]
    notes = []

    system_prompt = (
        "You are an academic research assistant teaching a beginner. "
        "Use only the supplied paper text. Do not invent facts, results, "
        "or limitations. Use simple, clear English."
    )

    for group_number, group in enumerate(groups, start=1):
        context = "\n\n".join(
            f"[PDF page {item['page']}]\n{item['text']}"
            for item in group
        )
        user_prompt = f"""
Summarize the following research paper passages in simple language.

Extract:
- The main idea or problem
- The method proposed
- Important results explicitly stated
- Limitations, only if explicitly stated

Use concise notes. Preserve numerical results and page references.
Do not guess when information is missing.

PAPER PASSAGES:
{context}
"""
        note = generate_text(
            tokenizer,
            model,
            system_prompt,
            user_prompt,
            max_new_tokens=200,
        )
        notes.append(note)
        progress_bar.progress(
            group_number / (len(groups) + 1),
            text=f"Summarizing section {group_number} of {len(groups)}...",
        )

    final_prompt = """
Write a simple, accurate summary of this research paper using
ONLY the notes below.

Use these headings:
1. Research Problem
2. Proposed Method
3. Main Findings
4. Limitations
5. Conclusion

Rules:
- Explain concepts in simple language for a beginner.
- Keep each section concise.
- Preserve numerical results exactly as stated in the notes.
- Do not invent facts, results, or training times.
- Include limitations only if the notes explicitly support them.
- Do not guess what the limitations might be.
- If limitations are not supported by the notes, write:
  "No explicit limitations were identified in the provided notes."
- If information is missing, say so.
- Include page references when available.

NOTES:
""" + "\n\n".join(notes)

    summary = generate_text(
        tokenizer,
        model,
        system_prompt,
        final_prompt,
        max_new_tokens=450,
    )
    progress_bar.progress(1.0, text="Summary complete!")
    return summary


uploaded_file = st.file_uploader(
    "Upload your research paper (PDF)",
    type=["pdf"],
)

if uploaded_file is None:
    st.info("Upload a PDF to get started.")
else:
    if uploaded_file.size > MAX_PDF_SIZE_MB * 1024 * 1024:
        st.error(
            f"This PDF is too large. The current limit is {MAX_PDF_SIZE_MB} MB."
        )
        st.stop()

    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        for key in (
            "file_hash",
            "chunks",
            "embeddings",
            "summary",
            "last_question",
            "last_answer",
            "last_sources",
        ):
            st.session_state.pop(key, None)

        try:
            with st.spinner("Reading PDF and creating embeddings..."):
                chunks = extract_chunks(pdf_bytes)
                if not chunks:
                    st.error("No readable text was found in this PDF.")
                    st.stop()

                embedder = load_embedding_model()
                embeddings = embedder.encode(
                    [item["text"] for item in chunks],
                    batch_size=16,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

                st.session_state["chunks"] = chunks
                st.session_state["embeddings"] = np.asarray(embeddings)
                st.session_state["file_hash"] = file_hash
        except Exception as error:
            st.error(f"Could not process this PDF: {error}")
            st.stop()

    chunks = st.session_state["chunks"]
    embeddings = st.session_state["embeddings"]
    st.success(f"Paper ready! Created {len(chunks)} text chunks.")

    st.divider()
    st.subheader("Summarize Paper")
    st.write("Generate a structured summary in simple language.")

    if st.button("Summarize Paper", type="primary"):
        progress_bar = st.progress(0, text="Preparing summary...")
        try:
            tokenizer, model = load_llm(LLM_MODEL, LLM_CPU_DTYPE)
            with st.spinner("Reading the paper and writing the summary..."):
                summary = summarize_paper(chunks, tokenizer, model, progress_bar)
            st.session_state["summary"] = summary
        except Exception as error:
            st.error(f"Could not summarize the paper: {error}")
        finally:
            progress_bar.empty()

    if st.session_state.get("summary"):
        st.markdown(st.session_state["summary"])
        st.download_button(
            "Download Summary",
            data=st.session_state["summary"],
            file_name="research_paper_summary.txt",
            mime="text/plain",
        )

    st.divider()
    st.subheader("Ask Your Paper")
    question = st.text_input(
        "What would you like to understand?",
        placeholder="What is self-attention? Explain simply.",
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Finding relevant passages..."):
                    embedder = load_embedding_model()
                    query_vector = embedder.encode(
                        question.strip(),
                        normalize_embeddings=True,
                    )
                    scores = embeddings @ query_vector
                    top_indices = np.argsort(scores)[::-1][:TOP_K]
                    retrieved = [
                        {**chunks[index], "score": float(scores[index])}
                        for index in top_indices
                    ]

                context = "\n\n".join(
                    f"[PDF page {item['page']}]\n{item['text']}"
                    for item in retrieved
                )
                tokenizer, model = load_llm(LLM_MODEL, LLM_CPU_DTYPE)
                system_prompt = (
                    "You are a friendly teacher explaining academic papers to "
                    "a beginner. Answer the exact question in simple everyday "
                    "English, normally in 2-3 short sentences. Use only the "
                    "supplied passages. Do not invent facts. Add PDF page "
                    "citations like [PDF page 2] when supported by the context. "
                    "If the passages do not answer the question, say so."
                )
                user_prompt = f"""
Question: {question}

Relevant passages from the research paper:
{context}

Explain the answer simply and directly. Include page citations
where supported by the passages.
"""
                with st.spinner("Writing your answer..."):
                    answer = generate_text(
                        tokenizer,
                        model,
                        system_prompt,
                        user_prompt,
                        max_new_tokens=150,
                    )

                st.session_state["last_question"] = question
                st.session_state["last_answer"] = answer
                st.session_state["last_sources"] = retrieved
            except Exception as error:
                st.error(f"Could not answer your question: {error}")

    if st.session_state.get("last_answer"):
        st.markdown("### Answer")
        st.write(st.session_state["last_answer"])
        st.markdown("### Sources from your paper")
        for rank, source in enumerate(st.session_state["last_sources"], start=1):
            with st.expander(
                f"Source {rank} — PDF page {source['page']} "
                f"(similarity: {source['score']:.3f})"
            ):
                st.write(source["text"])
