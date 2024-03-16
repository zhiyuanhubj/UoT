import sys 
# sys.path.append('gemma_pytorch')
from uot.gemma_pytorch.gemma.config import get_config_for_7b, get_config_for_2b
from uot.gemma_pytorch.gemma.model import GemmaForCausalLM
import contextlib
import os
import torch
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ['KAGGLE_CONFIG_DIR'] = '~/.kaggle'  # Replace '/path/to/.kaggle' with the actual path
api = KaggleApi()
api.authenticate()
kagglehub.login()

# # Load the model
VARIANT = "2b-it" 
MACHINE_TYPE = "cpu" 

# Load model weights
weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{VARIANT}')
print(weights_dir)
# weights_dir = 'kaggle/input/gemma/pytorch/7b/2' 

# Ensure that the tokenizer is present
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

# Ensure that the checkpoint is present
ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)

# Set up model config.
model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")
model_config.quant = 'quant' in VARIANT

device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
  model = GemmaForCausalLM(model_config)
  model.load_weights(ckpt_path)
  model = model.to(device).eval()


# Use the model

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"


def gemma_response(history, output_len):
    prompt = ""
    for h in history:
        if h["role"] == "user":
            prompt += USER_CHAT_TEMPLATE.format(prompt=h["content"])
        else:
            prompt += MODEL_CHAT_TEMPLATE.format(prompt=h["content"])
    prompt += "<start_of_turn>model\n"
    print(1)
    res = model.generate(
        USER_CHAT_TEMPLATE.format(prompt=prompt),
        device=device,
        output_len=output_len,
    )
    print(2)
    return res

# prompt = (
#     USER_CHAT_TEMPLATE.format(
#         prompt="What is a good place for travel in the US?"
#     )
#     + MODEL_CHAT_TEMPLATE.format(prompt="California.")
#     + USER_CHAT_TEMPLATE.format(prompt="What can I do in California?")
#     + "<start_of_turn>model\n"
# )

# res = model.generate(
#     USER_CHAT_TEMPLATE.format(prompt=prompt),
#     device=device,
#     output_len=100,
# )
# print(res)
