from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class ModelLoader:

    @staticmethod
    def load_model(model_path: str, bnb_config=None, precision: str = "fp16"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if precision not in DTYPE_MAP:
            raise ValueError(f"Unsupported precision '{precision}'. Choose from {list(DTYPE_MAP)}")

        dtype = DTYPE_MAP[precision]

        load_kwargs = {
            "device_map": device,
            "dtype": dtype,
        }
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
            # bitsandbytes int8 expects fp16 compute dtype — enforce it regardless of config,
            # otherwise you get the MatMul8bitLt cast-warning spam again
            if precision != "fp16":
                load_kwargs["dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        return model, tokenizer, device
# model , tokenizer, device = ModelLoader.load_model(model_path)

# prompt = "Write a poem about the beauty of nature."

# inputs = tokenizer(prompt, return_tensors="pt").to(device)

# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=100,
#     )

# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(generated_text)
# print(device)