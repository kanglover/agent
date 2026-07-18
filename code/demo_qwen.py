import gradio as gr
import os

from openai import OpenAI

api_key = os.getenv("DASHSCOPE_API_KEY")

def call_qwen(message, history):
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    messages = []
    if history:
        try:
            for msg in history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    messages.append(msg)
                elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                    user_msg, assistant_msg = msg
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
        except Exception as e:
            print(f"Error: {e}")

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="qwen-max",
            messages=messages,
            stream=False
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

demo = gr.ChatInterface(
    fn=call_qwen,
    title="Qwen",
    description="Qwen is a chatbot that can answer questions and provide information.",
)

demo.launch()