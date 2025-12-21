import gradio as gr
from . import retriever, llm

def create_interface():
    """Creates the Gradio interface for the Q&A assistant."""
    # Components
    question = gr.Textbox(label="Your Question", placeholder="Ask about Ultralytics YOLO code...")
    answer_box = gr.Markdown(label="Answer")
    code_box = gr.Markdown(label="Retrieved Code Snippets")
    
    # Define the response function
    def answer_question(query):
        try:
            chunks = retriever.get_relevant_chunks(query)
        except Exception as e:
            return f"*(Retriever error: {e})*", "*No relevant code context found.*"

        try:
            answer = llm.generate_answer(query, chunks)
        except Exception as e:
            return f"*(LLM error: {e})*", "*No relevant code context found.*"
        # Format retrieved chunks as collapsible HTML
        if chunks:
            details_html = "<details><summary>Show retrieved code context</summary>\n"
            for chunk in chunks:
                meta = chunk['meta']
                code_text = chunk['text']
                # Escape HTML characters in code text
                import html
                code_esc = html.escape(code_text)
                file_path = meta.get('path', 'unknown path')
                start_line = meta.get('start_line', 0)
                end_line = meta.get('end_line', 0)
                details_html += f"<pre><code class='language-python'># {file_path} L{start_line}-L{end_line}\n{code_esc}\n</code></pre>\n"
            details_html += "</details>"
        else:
            # If no chunks found, we inform user no context
            details_html = "*No relevant code context found.*"
        return answer, details_html
    
    # Set up the interface layout
    demo = gr.Interface(
        fn=answer_question,
        inputs=question,
        outputs=[answer_box, code_box],
        title="Ultralytics YOLO Code Assistant",
        description="""Ask questions about the Ultralytics YOLO codebase. The assistant will retrieve relevant code snippets and explain **what the code does** and **how to use it**, citing file names and line numbers."""
    )
    return demo

def main():
    # Create and launch the Gradio app
    interface = create_interface()
    interface.launch()
