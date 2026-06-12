from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# print("Downloaded successfully!")


import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerState, TrainerControl)

# Load the dataset
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
data = pd.read_csv("dialogs.txt", sep="\t", header=None, names=["input", "response"])
dataset = Dataset.from_pandas(data)

# Tokenize the dataset

def tokenize_function(examples):
    texts = []

    for inp, resp in zip(examples["input"], examples["response"]):
        text = f"Question: {inp}\nAnswer: {resp}"
        texts.append(text)

    model_inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # IMPORTANT: causal LM training
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


tokenized_dataset = dataset.map(tokenize_function, batched=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
#model 
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10,
    save_total_limit=2,
    logging_steps=10,
)

# train the model
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_dataset,
)

trainer.train()

trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")




