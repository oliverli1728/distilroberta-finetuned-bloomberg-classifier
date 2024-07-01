import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
from transformers import pipeline

pipe = pipeline("text-classification", model="oli1728/distilroberta-finetuned-bloomberg-classifier")

def predict(prompt):
    completion = pipe(prompt)
    return completion

import gradio as gr

gr.Interface(fn=predict, inputs="text", outputs="text").launch()