"""
app.py
Gradio web UI for the CTI-RAG Threat Intelligence Q&A pipeline.
Run: python app.py
"""

import gradio as gr
from pathlib import Path
from pipeline import RAGPipeline, load_config


cfg = load_config()
pipeline = RAGPipeline(cfg)


def process_document(file_obj) -> str:
    if file_obj is None:
        return "No file uploaded."
    try:
        pipeline.index_document(file_obj.name)
        chunk_count = len(pipeline.retriever.chunks)
        return f"✅ Document indexed successfully! {chunk_count} chunks created. Ready to answer questions."
    except Exception as e:
        return f"❌ Error processing document: {str(e)}"


def index_from_url(url: str) -> str:
    if not url.strip():
        return "No URL provided."
    try:
        pipeline.index_document(url.strip())
        chunk_count = len(pipeline.retriever.chunks)
        return f"✅ Indexed from URL! {chunk_count} chunks created."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def answer_question(question: str, show_sources: bool) -> tuple[str, str, str]:
    if not question.strip():
        return "Please enter a question.", "", ""
    if not pipeline._is_ready:
        return "Please upload and index a document first.", "", ""

    try:
        result = pipeline.ask(question, verbose=False)
        answer = result["answer"]

        # Format IOCs
        ioc_text = ""
        iocs = result.get("iocs", {})
        if iocs:
            parts = []
            for ioc_type, values in iocs.items():
                parts.append(f"**{ioc_type.upper()}**: {', '.join(values)}")
            ioc_text = "\n\n".join(parts)
        else:
            ioc_text = "_No IOCs detected in this response._"

        # Format sources
        sources = ""
        if show_sources:
            sources_parts = []
            for chunk in result["retrieved_chunks"]:
                src = chunk.get("source", "unknown")
                score = chunk.get("score", 0)
                sources_parts.append(
                    f"**Source: {src}** (relevance: {score:.3f})\n"
                    f"{chunk['text'][:300]}..."
                )
            sources = "\n\n---\n\n".join(sources_parts)

        return answer, ioc_text, sources
    except Exception as e:
        return f"Error: {str(e)}", "", ""


with gr.Blocks(title="🛡️ CTI Threat Intelligence Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🛡️ CTI Threat Intelligence Q&A
    Upload a threat report (PDF/TXT/JSON) or paste a URL to a CISA advisory, then ask security-specific questions.
    Built with sentence-transformers + FAISS + Gemini.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Document")
            file_input = gr.File(
                label="Upload PDF, TXT, or JSON CVE feed",
                file_types=[".pdf", ".txt", ".json"],
            )
            upload_btn = gr.Button("📥 Index Document", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### — OR — Index from URL")
            url_input = gr.Textbox(
                label="Threat report URL",
                placeholder="https://www.cisa.gov/...",
                lines=1,
            )
            url_btn = gr.Button("🌐 Index from URL", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("### 2. Ask Questions")
            question_input = gr.Textbox(
                label="Analyst question",
                placeholder="What TTPs does APT29 use for lateral movement?...",
                lines=2,
            )
            show_sources = gr.Checkbox(label="Show retrieved source chunks", value=True)
            ask_btn = gr.Button("🔍 Get Assessment", variant="primary")

            answer_output = gr.Textbox(label="Intelligence Assessment", lines=4, interactive=False)
            ioc_output = gr.Markdown(label="Extracted IOCs", visible=True)
            sources_output = gr.Markdown(label="Retrieved Sources", visible=True)

    upload_btn.click(fn=process_document, inputs=file_input, outputs=upload_status)
    url_btn.click(fn=index_from_url, inputs=url_input, outputs=upload_status)
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, show_sources],
        outputs=[answer_output, ioc_output, sources_output],
    )

    gr.Examples(
        examples=[
            ["What TTPs does APT29 use for lateral movement?"],
            ["Which CVEs in this advisory affect Windows Server?"],
            ["What IOCs are associated with the Lazarus Group?"],
            ["What MITRE ATT&CK techniques are referenced in this report?"],
        ],
        inputs=question_input,
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
