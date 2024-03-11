import gradio as gr
from fow import ChatAgent, EXAMPLES

chatter = ChatAgent("SONY", "data/freelancer.json", "mistralai/Mistral-7B-v0.1", load_in_8bit=True)
chatter.interface()