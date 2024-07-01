import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
from datasets import Dataset
from datasets import DatasetDict
from transformers import pipeline
from transformers import (
    TrainingArguments,
    Trainer,
)
from huggingface_hub import HfFolder

from transformers import AutoTokenizer, AutoModelForSequenceClassification

sentiment_pipeline = pipeline("sentiment-analysis")

import pandas as pd

df = pd.read_parquet("hf://datasets/Danbnyn/Bloomberg_Financial_News/bloomberg_financial_data.parquet.gzip")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
df = df.drop(columns=["Link", "Headline", "Journalists", "Date"])
df = df.iloc[:12500, :]
list_label= list()
list_text = df["Article"].tolist()
for i in range (12500):
    tokenized_sentence = tokenizer.encode(df.loc[i, "Article"], padding=True, truncation=True, max_length=510, add_special_tokens = True)
    decoded = tokenizer.decode(tokenized_sentence)
    list_label.append(sentiment_pipeline(decoded)[0]["label"])

list_text_train = list_text[7500:10000]
list_text_eval = list_text[10000:12500]

list_label_train = list_label[7500:10000]
list_label_eval = list_label[10000:12500]
train_df = pd.DataFrame({
     "label" : list_label_train,
     "Article" : list_text_train
})
eval_df = pd.DataFrame({
     "label" : list_label_eval,
     "Article" : list_text_eval
})
eval_df['label'] = eval_df['label'].replace(['NEGATIVE','POSITIVE'],[0,1])
train_df['label'] = train_df['label'].replace(['NEGATIVE','POSITIVE'],[0,1])
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
ds = DatasetDict()
ds['train'] = train_dataset
ds['validation'] = eval_dataset

tokenizer = AutoTokenizer.from_pretrained("oli1728/distilroberta-finetuned-bloomberg-classifier")
model = AutoModelForSequenceClassification.from_pretrained("oli1728/distilroberta-finetuned-bloomberg-classifier")

def tokenize_func(dataset):
    return tokenizer(dataset['Article'], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize_func, batched=True, batch_size=len(train_dataset))
eval_dataset = eval_dataset.map(tokenize_func, batched=True, batch_size=len(eval_dataset))
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])



training_args = TrainingArguments(
    output_dir="oli1728/distilroberta-finetuned-bloomberg-classifier",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="oli1728/distilroberta-finetuned-bloomberg-classifier/logs",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id="oli1728/distilroberta-finetuned-bloomberg-classifier",
    hub_token=HfFolder.get_token(),
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
trainer.evaluate()
trainer.save_model("./my_model")
tokenizer.save_pretrained("oli1728/distilroberta-finetuned-bloomberg-classifier")
trainer.push_to_hub("oli1728/distilroberta-finetuned-bloomberg-classifier")
