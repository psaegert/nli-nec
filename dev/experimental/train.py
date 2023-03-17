import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from nlinec import combine_positive_negative_data, combine_premise_hypothesis, construct_hypothesis, get_all_types, get_granularity, get_models_dir, get_negative_data, get_positive_data, get_type
from nlinec.predict import predict_type

GRANULARITY = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.path.join(get_models_dir(), f'nlinec-{GRANULARITY}-logging')


positive_data = get_positive_data("augmented_train.json", explode=True)
negative_data = get_negative_data("augmented_train.json", random_state=42)
data = combine_positive_negative_data(positive_data, negative_data, frac=0.5, random_state=42)
dev_data = get_positive_data("g_dev.json", explode=True)

del positive_data, negative_data

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


gran_types = []
for i in [1, 2, 3]:
    all_types = get_all_types(granularity=i)
    all_types['granularity'] = all_types['full_type'].apply(lambda x: get_granularity(x))
    gran_types.append(all_types[all_types['granularity'] == i])


tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(DEVICE)


def compute_dev_accuracy(model, tokenizer, dev_data, types):  # type: ignore
    with torch.no_grad():
        dev_predictions = predict_type(model, tokenizer, list(dev_data['sentence']), list(dev_data['mention_span']), list(types), return_str=True, verbose=True)
    return (dev_data[f'type_{GRANULARITY}'] == dev_predictions).mean()


def tokenize_function(examples: dict) -> dict:
    # input_text = examples["sentence"] + "</s><s>" + examples["hypothesis"]
    input_text = [combine_premise_hypothesis(sentence, hypothesis) for sentence, hypothesis in zip(examples["sentence"], examples["hypothesis"])]
    return tokenizer(input_text, max_length=model.config.max_position_embeddings, padding="max_length", return_tensors="pt")


class AccuracyCallback(TrainerCallback):
    def __init__(self, dev_data: pd.DataFrame, types: list, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, steps: list[int] = None):
        self.model = model
        self.tokenizer = tokenizer

        self.dev_data = dev_data
        self.types = types

        self.accuracy: dict = {}

        self.steps = steps if steps is not None else []

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: dict) -> None:
        if state.global_step in self.steps or state.global_step == state.max_steps:
            # Compute the dev accuracy with the current model
            accuracy = compute_dev_accuracy(self.model, self.tokenizer, self.dev_data, self.types)

            loss = state.log_history[-1]['loss'] if len(state.log_history) > 0 else -1
            print(f"Accuracy: {accuracy:.4f} Loss: {loss:.4f} Step: {state.global_step}")

            # Save the accuracy
            self.accuracy[state.global_step] = accuracy


# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
train_dataset = Dataset.from_pandas(data.loc[:, ["sentence", "hypothesis", "label"]])
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

# Create an instance of the callback
callback_steps = np.concatenate([
    np.arange(0, 10, 1),
    np.arange(10, 50, 5),
    np.arange(50, 100, 10),
    np.arange(100, 500, 50),
    np.arange(500, 1000, 100),
    np.arange(1000, 5000, 500),
    np.arange(5000, 10000, 1000),
    np.arange(10000, 50000, 5000)],
    axis=None)
print(f'Callback steps: {len(callback_steps)}')
accuracy_callback = AccuracyCallback(dev_data, list(gran_types[1]['type']), model, tokenizer, steps=callback_steps.tolist())

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    warmup_steps=500,
    logging_dir='./logs',
    save_steps=1e10,
    logging_steps=1,
    gradient_accumulation_steps=4,
    evaluation_strategy='no',

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    callbacks=[accuracy_callback],
)

try:
    # Train the model
    trainer.train()
except KeyboardInterrupt:
    # Save the accuracy
    print('Saving the accuracy')
    with open(os.path.join(OUTPUT_DIR, "accuracy.json"), "w") as f:
        json.dump(accuracy_callback.accuracy, f)

    # Save the loss
    print('Saving the loss')
    with open(os.path.join(OUTPUT_DIR, "loss.json"), "w") as f:
        json.dump(trainer.state.log_history, f)

# Save the model
trainer.save_model(OUTPUT_DIR)
