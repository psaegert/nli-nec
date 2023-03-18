import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .data.load import combine_positive_negative_data, get_all_types, get_granularity, get_negative_data, get_positive_data, get_type
from .data.preprocessing import combine_premise_hypothesis, construct_hypothesis
from .predict import predict_type
from .utils import get_models_dir


def compute_accuracy(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, data: pd.DataFrame, types: list, granularity: int) -> float:
    """
    Compute the accuracy on the data.

    Parameters
    ----------
    model: PreTrainedModel
        The model to use for prediction.
    tokenizer: PreTrainedTokenizer
        The tokenizer to use for prediction.
    data: pd.DataFrame
        The data to predict on.
    types: list
        The list of types to choose from.
    granularity: int, {1, 2, 3}
        The granularity of the types to compute the accuracy on.

    Returns
    -------
    float
        The accuracy on the data.
    """
    with torch.no_grad():
        predictions = predict_type(model, tokenizer, list(data['sentence']), list(data['mention_span']), types, return_str=True, verbose=True)
    return (data[f'type_{granularity}'] == predictions).mean()


def generate_log_callback_steps(max_step: float) -> set:
    log_10_max_step = int(1 + np.log10(max_step))
    steps = [0, 1, 2, 3, 4]
    for magnitude in range(1, log_10_max_step):
        steps.extend(np.arange(5 * 10**(magnitude - 1), 10**magnitude, 10**(magnitude - 1)))
        steps.extend(np.arange(1 * 10**magnitude, 5 * 10**magnitude, 5 * 10**(magnitude - 1)))
    steps_array = np.array(steps)
    return set(steps_array[steps_array <= max_step])


class AccuracyCallback(TrainerCallback):
    """
    A callback to compute the accuracy on the dev set during training and save it to a json file.
    """
    def __init__(self, dev_data: pd.DataFrame, types: list, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str, granularity: int, steps: list[int] | set[int] | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.granularity = granularity

        self.dev_data = dev_data
        self.types = types

        self.accuracy: dict = {}
        self.steps = steps or set()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: dict) -> None:
        if state.global_step in self.steps or state.global_step == state.max_steps:
            # Compute the dev accuracy with the current model
            accuracy = compute_accuracy(self.model, self.tokenizer, self.dev_data, self.types, self.granularity)
            loss = state.log_history[-1]['loss'] if len(state.log_history) > 0 else -1

            print(f"Accuracy: {accuracy:.4f} Loss: {loss:.4f} Step: {state.global_step}")

            # Save the accuracy
            self.accuracy[state.global_step] = accuracy

            # Save the accuracy
            print('Saving the accuracy')
            with open(os.path.join(self.output_dir, "accuracy.json"), "w") as f:
                json.dump(self.accuracy, f)

            # Save the loss
            print('Saving the loss')
            with open(os.path.join(self.output_dir, "loss.json"), "w") as f:
                json.dump(state.log_history, f)


def train(model_name: str, granularity: int, device: str | None = None, negative_frac: float = 0.5, random_state: int | None = None, train_frac: float = 1) -> None:
    """
    Train the `roberta-large-mnli` model on the augmented ontonotes NEC dataset and save the model and its logs.

    Parameters
    ----------
    model_name: str
        The name used to save the model.
    granularity: int, {1, 2, 3}
        The granularity of the types to train on.
    device: str, {"cuda", "cpu"}
        The device to train on.
    negative_frac: float, optional
        The fraction of negative data to use, by default 0.5.
    random_state: int, optional
        The random state to use for the negative data, by default None.
    train_frac: int, optional
        The fraction of the training data to use, by default 1.
    """
    # If the device is not specified, use cuda if available
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-using device: {device}")

    # --- Load ---

    # Load the pre-trained model and tokenizer for the roberta-large-mnli model
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)

    # Get the path to save the model
    output_dir = os.path.join(get_models_dir(), model_name)

    # Load the data
    positive_data = get_positive_data("augmented_train.json", explode=True)
    negative_data = get_negative_data("augmented_train.json", random_state=random_state)

    # Combine the positive and negative data of the combination fraction is not 0
    if negative_frac != 0:
        data = combine_positive_negative_data(positive_data, negative_data, frac=0.5, random_state=random_state)
    else:
        data = positive_data

    # Remove the original data to save memory
    del positive_data, negative_data

    # Load the dev data for validation and accuracy logging during training
    dev_data = get_positive_data("g_dev.json", explode=True)

    # --- /Load ---

    # --- Preprocessing ---

    # Add the basic type
    data[f'type_{granularity}'] = data['full_type'].apply(lambda x: get_type(x, granularity))
    dev_data[f'type_{granularity}'] = dev_data['full_type'].apply(lambda x: get_type(x, granularity))

    # Remove the rows with type None
    data = data[data[f'type_{granularity}'].notna()]
    dev_data = dev_data[dev_data[f'type_{granularity}'].notna()]

    # Remove duplicates
    data = data.drop_duplicates(subset=['mention_span', 'sentence', f'type_{granularity}'])
    dev_data = dev_data.drop_duplicates(subset=['mention_span', 'sentence', f'type_{granularity}'])

    # Construct the hypothesis
    data["hypothesis"] = data.apply(lambda row: construct_hypothesis(row["mention_span"], row[f'type_{granularity}']), axis=1)
    dev_data["hypothesis"] = dev_data.apply(lambda row: construct_hypothesis(row["mention_span"], row[f'type_{granularity}']), axis=1)

    def tokenize_function(examples: dict) -> dict:
        input_text = [combine_premise_hypothesis(sentence, hypothesis) for sentence, hypothesis in zip(examples["sentence"], examples["hypothesis"])]
        return tokenizer(input_text, max_length=model.config.max_position_embeddings, padding="max_length", return_tensors="pt")

    # Shuffle, sample, and tokenize the data
    data = data.sample(frac=train_frac, random_state=random_state).reset_index(drop=True)
    train_dataset = Dataset.from_pandas(data.loc[:, ["sentence", "hypothesis", "label"]])
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

    # --- /Preprocessing ---

    # --- Setup ---

    # Create an instance of the callback
    callback_steps = generate_log_callback_steps(len(tokenized_train_dataset) // (8 * 8))  # 8 is the batch size, 8 is the number of gradient accumulation steps
    print(f'Callback Steps: {sorted(list(callback_steps))}')

    # Get the types at the specified granularity
    all_types = get_all_types(granularity=granularity)
    all_types['granularity'] = all_types['full_type'].apply(lambda x: get_granularity(x))
    gran_types = all_types[all_types['granularity'] == granularity]

    # Create an instance of the callback
    accuracy_callback = AccuracyCallback(
        dev_data,
        list(gran_types['type']),
        model,
        tokenizer,
        output_dir=output_dir,
        granularity=granularity,
        steps=callback_steps)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-6,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        warmup_steps=500,
        logging_dir='./logs',
        save_steps=1e10,
        logging_steps=1,
        gradient_accumulation_steps=8,
        evaluation_strategy='no')

    # Create an instance of the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        callbacks=[accuracy_callback])

    # --- /Setup ---

    # --- Training ---
    try:
        # Train the model
        trainer.train()
    except KeyboardInterrupt:
        # If the training is interrupted, save the model
        pass

    # Save the model
    trainer.save_model(output_dir)

    # --- /Training ---
