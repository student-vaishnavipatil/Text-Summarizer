import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = ("../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff")
text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

# text = "Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman and investor known for his key roles in the space company SpaceX and the automotive company Tesla, Inc. Other involvements include ownership of X Corp., the company that operates the social media platform X (formerly known as Twitter), and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. He is one of the wealthiest individuals in the world; as of August 2024 Forbes estimates his net worth to be US$247 billion.[3]"
# print(text_summary(text));

def summary (input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

demo = gr.Interface(fn=summary,
                    inputs=[gr.Textbox(label="Input text to summarize",lines=6)],
                    outputs=[gr.Textbox(label="Summarized text",lines=4)],
                    title="@GenAILearniverse Project 1: Text Summarizer",
                    description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE TEXT")
demo.launch()

