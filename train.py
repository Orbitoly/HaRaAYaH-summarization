from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer
import numpy as np
import nltk
import torch
import os
import huggingface_hub
nltk.download('punkt')

if torch.cuda.is_available():
    print("CUDA IS AVAILABLE")
else: print("CUDA IS NOT AVAILABLE")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
api_key = "SECRET_TOKEN" #Or's token
huggingface_hub.login(api_key)

model_checkpoint = "google/mt5-small"
dataset_name = "NLP-MINI-PROJECT/rabbi_kook"
#dataset_name = "imvladikon/hebrew_news" #NLP-MINI-PROJECT/rabbi_kook"
raw_datasets = load_dataset(dataset_name)
#raw_datasets = raw_datasets['train'].train_test_split(test_size=0.975)

metric = evaluate.load('rouge')
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

prefix = ""
max_input_length = 4096
max_target_length = 1024

raw_text_column_name = "paragraph"#"articleBody"
summary_column_name = "summary"#"description"

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples[raw_text_column_name]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[summary_column_name], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

batch_size = 1

model_name = model_checkpoint.split("/")[-1]
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print("DEVICE: ", device)
torch.cuda.empty_cache()
import gc
gc.collect()

data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

args = Seq2SeqTrainingArguments(
    f"{model_name}-kook-summary-or",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_strategy="no",
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(trainer)
trainer.train()
print("AFTER TRAIN :)")
trainer.save_model()
print("SAVED MODEL LOCALLY :)")

trainer.push_to_hub()
print("PUSHED TO HUB :)")