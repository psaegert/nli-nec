# %%
import os

import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from nlinec import combine_premise_hypothesis, construct_hypothesis, get_all_types, get_granularity, get_models_dir, get_positive_data, get_type
from nlinec.predict import predict_type

# %%
GRANULARITY = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
data = get_positive_data("augmented_train.json", explode=True)
dev_data = get_positive_data("g_dev.json", explode=True)

# %%
# Add the basic type
data[f'type_{GRANULARITY}'] = data['full_type'].apply(lambda x: get_type(x, GRANULARITY))
dev_data[f'type_{GRANULARITY}'] = dev_data['full_type'].apply(lambda x: get_type(x, GRANULARITY))

# Remove the rows with type None or "other"
data = data[data[f'type_{GRANULARITY}'].notna()]
dev_data = dev_data[dev_data[f'type_{GRANULARITY}'].notna()]
# data = data[data[f'type_{GRANULARITY}'] != 'other']

# Remove duplicates
data = data.drop_duplicates(subset=['mention_span', 'sentence', f'type_{GRANULARITY}'])
dev_data = dev_data.drop_duplicates(subset=['mention_span', 'sentence', f'type_{GRANULARITY}'])

# Construct the hypothesis
data["hypothesis"] = data.apply(lambda row: construct_hypothesis(row["mention_span"], row[f'type_{GRANULARITY}']), axis=1)
dev_data["hypothesis"] = dev_data.apply(lambda row: construct_hypothesis(row["mention_span"], row[f'type_{GRANULARITY}']), axis=1)


# %%
gran_types = []
for i in [1, 2, 3]:
    all_types = get_all_types(granularity=i)
    all_types['granularity'] = all_types['full_type'].apply(lambda x: get_granularity(x))
    gran_types.append(all_types[all_types['granularity'] == i])

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(DEVICE)

# %%
with torch.no_grad():
    dev_predictions = predict_type(model, tokenizer, list(dev_data['sentence']), list(dev_data['mention_span']), list(gran_types[1]['type']), return_str=True, verbose=True)

# %%
dev_data['prediction_before'] = dev_predictions

# %%
print(f"Dev accuracy: {(dev_data[f'type_{GRANULARITY}'] == dev_data['prediction_before']).mean()}")


# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)
train_dataset = Dataset.from_pandas(data.loc[:, ["sentence", "hypothesis", "label"]])


def tokenize_function(examples: dict) -> dict:
    # input_text = examples["sentence"] + "</s><s>" + examples["hypothesis"]
    input_text = [combine_premise_hypothesis(sentence, hypothesis) for sentence, hypothesis in zip(examples["sentence"], examples["hypothesis"])]
    return tokenizer(input_text, max_length=model.config.max_position_embeddings, padding="max_length", return_tensors="pt")


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

# %%
output_dir = os.path.join(get_models_dir(), f'nlinec-positive-{GRANULARITY}')


training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    # per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=3000,
    gradient_accumulation_steps=4,
    save_steps=3000,
    # load_best_model_at_end=True,
    # evaluation_strategy="steps",
    evaluation_strategy='no',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    # eval_dataset=tokenized_val_dataset,
    # compute_metrics=compute_metrics,
)

# %%
# Train the model
trainer.train()

# %%
# Save the model
trainer.save_model(output_dir)

# %%
with torch.no_grad():
    dev_predictions = predict_type(model, tokenizer, list(dev_data['sentence']), list(dev_data['mention_span']), list(gran_types[1]['type']), return_str=True, verbose=True)

# %%
dev_data['prediction_after'] = dev_predictions

# %%
print(f"Dev accuracy after training: {(dev_data[f'type_{GRANULARITY}'] == dev_data['prediction_after']).mean()}")
