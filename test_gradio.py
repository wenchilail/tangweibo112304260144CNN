import gradio as gr

def greet(name):
    return f"Hello {name}!"

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="Test",
    description="Simple gradio test",
)

if __name__ == "__main__":
    print("Starting Gradio test...")
    demo.launch(server_name="0.0.0.0", server_port=7861)
