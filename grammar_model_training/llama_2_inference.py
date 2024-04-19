from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
import torch
import pandas as pd
import os
import json


device_map = {"": 0}

# Reload the Llama2 model
model_name = "./merged_llama/final_merged_checkpoint/"
# model_name = "./merged_dpo/final_merged_checkpoint/"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# DATA_PATH = "/local-scratch1/data/siyanli/gec_feedback/dataset/annotated/"
test_data = pd.read_csv(os.path.join("data/", "test.csv"))

logging.set_verbosity(logging.CRITICAL)

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=400)
output_inferences = []

for i, a in test_data.iterrows():
    if int(i) > 100:
        break
    result = pipe(f"<s>[INST] Correct the grammar in the following sentence: {a['original']} [/INST]")

    output_inferences.append({
        "prompt": a["original"],
        "og_completion": a["corrected"],
        "inference": result[0]['generated_text']
    })
json.dump(output_inferences, open("runtime_inference_llama.json", "w+"))