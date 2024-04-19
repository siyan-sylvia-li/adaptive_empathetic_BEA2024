"""
    Code adapted from the peft-flan-t5-int8-summarization notebook
"""
import argparse
import json
import pickle
import datasets
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer
import os


def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = ["Correct the grammar in the following sentence: " + item for item in sample["original"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=False)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["corrected"], max_length=max_target_length, padding=padding, truncation=False)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/flan-t5-xl")
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="grammar_bot_xl")
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    DATA_PATH = "data/"
    training_data = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    eval_data = pd.read_csv(os.path.join(DATA_PATH, "dev.csv"))
    test_data = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=training_data))
    val_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=eval_data))
    test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=test_data))

    max_source_length, max_target_length = 512, 512
    model_id = args.model

    # Load tokenizer of FLAN-t5-Large
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")


    # Preprocess dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["original", "corrected"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset.features)}")
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["original", "corrected"])
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=["original", "corrected"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset.features)}")



    """
        Load Model
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

    from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    from transformers import DataCollatorForSeq2Seq

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    output_dir = args.output_dir + "-exp"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=args.lr,  # higher learning rate
        num_train_epochs=args.max_epoch,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["tensorboard"],
        weight_decay=1e-9,
        evaluation_strategy="epoch",
        load_best_model_at_end=True
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_val_dataset
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # train model
    trainer.train()

    trained_model_id = args.output_dir
    trainer.model.save_pretrained(trained_model_id)
    tokenizer.save_pretrained(trained_model_id)