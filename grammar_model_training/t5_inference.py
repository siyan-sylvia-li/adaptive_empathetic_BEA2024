from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse
import json
import pickle
import datasets
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer
import os


if __name__ == "__main__":
    max_source_length, max_target_length = 512, 512
    output_inferences = []
    tokenizer = T5Tokenizer.from_pretrained('grammar_bot_xl')
    # DATA_PATH = "/local-scratch1/data/siyanli/gec_feedback/dataset/annotated/"
    test_data = pd.read_csv(os.path.join("data/", "test.csv"))
    # test_data = pd.read_csv("output.csv")
    model = AutoModelForSeq2SeqLM.from_pretrained("grammar_bot_xl", device_map="cuda:0")
    for i, a in test_data.iterrows():
        # Only assess the first 100 examples
        if int(i) > 100:
            break
        token = tokenizer("Correct the grammar in the following sentence: " + a["original"], return_tensors="pt")

        input_ids = token["input_ids"]
        attention_mask = token["attention_mask"]

        # 'set num_beams = 1' for greedy search
        tokens = model.generate(input_ids=input_ids.to("cuda:0"), attention_mask=attention_mask.to("cuda:0"), num_beams=2, max_new_tokens=512)

        output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)

        output_inferences.append({
            "prompt": a["original"],
            "og_completion": a["corrected"],
            "inference": output
        })
    json.dump(output_inferences, open("grammar_bot_inference_xl.json", "w+"))
