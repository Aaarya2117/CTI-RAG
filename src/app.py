"""
app.py
Gradio web UI for the RAG Document Q&A pipeline.
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


def answer_question(question: str, show_sources: bool) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""

    if not pipeline._is_ready:
        return "Please upload and index a document first.", ""

    try:
        result = pipeline.ask(question, verbose=False)
        answer = result["answer"]

        sources = ""
        if show_sources:
            sources_parts = []
            for chunk in result["retrieved_chunks"]:
                sources_parts.append(
                    f"**Chunk {chunk['rank']} (relevance: {chunk['score']:.3f})**\n"
                    f"{chunk['text'][:300]}..."
                )
            sources = "\n\n---\n\n".join(sources_parts)

        return answer, sources
    except Exception as e:
        return f"Error: {str(e)}", ""


with gr.Blocks(title="RAG Document Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📄 RAG Document Q&A
    Upload a PDF or TXT document, then ask questions about it.
    Built with sentence-transformers + FAISS + Flan-T5.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Document")
            file_input = gr.File(
                label="Upload PDF or TXT",
                file_types=[".pdf", ".txt"],
            )
            upload_btn = gr.Button("📥 Index Document", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 2. Ask Questions")
            question_input = gr.Textbox(
                label="Your question",
                placeholder="What is the main topic? What methods are used?...",
                lines=2,
            )
            show_sources = gr.Checkbox(label="Show retrieved source chunks", value=True)
            ask_btn = gr.Button("🔍 Get Answer", variant="primary")

            answer_output = gr.Textbox(label="Answer", lines=4, interactive=False)
            sources_output = gr.Markdown(label="Retrieved Sources", visible=True)

    upload_btn.click(fn=process_document, inputs=file_input, outputs=upload_status)
    ask_btn.click(fn=answer_question, inputs=[question_input, show_sources], outputs=[answer_output, sources_output])

    gr.Examples(
        examples=[
            ["What is the main topic of this document?"],
            ["What methods or approaches are described?"],
            ["What are the key conclusions or findings?"],
        ],
        inputs=question_input,
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
