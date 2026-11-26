import json
import time
import gradio as gr
from aphelion_judge import AphelionJudge

judge = AphelionJudge(model_id="knss12/Aphelion-Judge-3.8B", backend="gguf")


def verify_ui(context: str, question: str, response: str):
    if not context.strip() or not response.strip():
        return {"reason": "Context or Response is empty.", "grounded": False}, "0 ms"
    t0 = time.perf_counter()
    verdict = judge.verify(context=context, response=response, question=question)
    ms = (time.perf_counter() - t0) * 1000.0
    return verdict, f"{ms:.1f} ms"


with gr.Blocks(title="Aphelion-Judge-3.8B") as demo:
    gr.Markdown("# Aphelion-Judge-3.8B — RAG Groundedness Verifier")
    with gr.Row():
        with gr.Column(scale=3):
            ctx = gr.Textbox(lines=6, label="Context")
            q = gr.Textbox(lines=1, label="Question (Optional)")
            resp = gr.Textbox(lines=3, label="AI Response")
            btn = gr.Button("Verify Groundedness", variant="primary")
        with gr.Column(scale=2):
            out = gr.JSON(label="Verdict")
            lat = gr.Textbox(label="CPU Latency", interactive=False)
    btn.click(verify_ui, [ctx, q, resp], [out, lat], api_name="verify")

if __name__ == "__main__":
    demo.launch()
