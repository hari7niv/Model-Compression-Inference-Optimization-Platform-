import argparse

import torch
from transformers import BitsAndBytesConfig

from ConfigLoader import ConfigLoader
from ModelLoader import ModelLoader

parser = argparse.ArgumentParser()

parser.add_argument("--config", required=True, help="Path to the configuration file")

args = parser.parse_args()

config = ConfigLoader.load(args.config)

model_path = config["model_path"]

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model, tokenizer, device = ModelLoader.load_model(model_path, bnb_config)


for name, module in model.named_modules():
    if "Linear8bitLt" in str(type(module)):
        print(f"✅ Quantized layer found: {name} — {type(module).__name__}")
        break
else:
    print("❌ No Linear8bitLt layers found — quantization did not apply")

print(f"Model memory footprint: {model.get_memory_footprint() / (1024**2):.1f} MB")