from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
import torch

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoTokenizer, PretrainedConfig
from transformers import LlamaModel


from safetensors.torch import load_file

from typing import List

import argparse

import gc

import os



parser = argparse.ArgumentParser(description="Simple example of a training script.")

parser.add_argument(
    "--projection_path",
    type=str,
    default="projection.safetensors",
    required=False,
    help="Path to pretrained projection",
)

parser.add_argument(
    "--pool_projection_path",
    type=str,
    default="pool_projection.safetensors",
    required=False,
    help="Path to pretrained pool projection",
)

parser.add_argument(
    "--unet_path",
    type=str,
    default="unet",
    required=False,
    help="Path to pretrained unet",
)

parser.add_argument(
    "--encoder",
    type=str,
    default=None,
    required=True,
    help="encoder repo or path",
)

parser.add_argument(
    "--pipeline_base",
    type=str,
    default="recoilme/colorfulxl",
    required=False,
    help="Path to pipeline base",
)


parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    required=True,
    help="prompt",
)

parser.add_argument(
    "--images_count",
    type=int,
    default=1,
    required=False,
    help="number of images to be generated",
)

parser.add_argument(
    "--output_path",
    type=str,
    default="out",
    required=False,
    help="path to save images",
)

args = parser.parse_args()

class AdaLayerNorm(torch.nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class ProjLayer(torch.nn.Module):
    def init(self, llama_config):
        super().__init__()
        self.norm_0 = AdaLayerNorm(llama_config.hidden_size)
        self.transform = torch.nn.TransformerDecoderLayer(llama_config.hidden_size, 8, llama_config.hidden_size, batch_first=True)
        self.norm_1 = AdaLayerNorm(llama_config.hidden_size)


    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor, emb: torch.Tensor):
        normed_hidden_states = self.norm_0(hidden_states, emb)
        normed_x = self.norm_1(x, emb)
        hidden_states = hidden_states + self.transform(normed_hidden_states, normed_x)
        return hidden_states

class PoolProjection(torch.nn.Module):
    def __init__(self, llama_config, out_dim=1280, num_layers=4):
        self.layers = torch.nn.ModuleList([ProjLayer(llama_config) for i in range(num_layers)])
        self.out = torch.nn.LazyLinear(out_dim)


    def forward(self, x: torch.Tensor, timestep: torch.Tensor):
        states = x
        emb = self.emb(timestep)
        for layer in self.layers:
            states = layer(x, states, emb)
        states = torch.nn.Flatten()(states)
        return self.out(states)

class Projection(torch.nn.Module):

    def __init__(self, llama_config, out_dim=1280, num_layers=4):
        self.layers = torch.nn.ModuleList([ProjLayer(llama_config) for i in range(num_layers)])
        self.out = torch.nn.LazyLinear(out_dim)


    def forward(self, x: torch.Tensor, timestep: torch.Tensor):
        states = x
        emb = self.emb(timestep)
        for layer in self.layers:
            states = layer(x, states, emb)
        return self.out(states)


device = "cuda" if torch.cuda.is_available() else "cpu"
text_encoder = LlamaModel.from_pretrained(args.encoder)
tokenizer = AutoTokenizer.from_pretrained(args.encoder)
tokenizer.pad_token_id = 0
projection_state_dict = load_file(args.projection_path)

pool_projection_state_dict = load_file(args.pool_projection_path)


projection = Projection(text_encoder.config, 2048)
pool_projection = PoolProjection(text_encoder.config, 1280)

projection.load_state_dict(projection_state_dict)
pool_projection.load_state_dict(pool_projection_state_dict)

def encode_prompt(prompt: List[str], tokenizer: AutoTokenizer, encoder: LlamaModel, projection: Projection, pool_projection: PoolProjection):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_encoder_embeds = text_encoder(
        text_input_ids.to(text_encoder.device),
        output_hidden_states=True,
        return_dict=True,
    ).last_hidden_state
    prompt_embeds = projection(text_encoder_embeds)
    pooled_prompt_embeds = pool_projection(prompt_embeds)
    return prompt_embeds, pooled_prompt_embeds


projection = projection.to(device)
pool_projection = pool_projection.to(device)
text_encoder = text_encoder.to(device)
prompt_embeds, pooled_prompt_embeds = encode_prompt([args.prompt], tokenizer, text_encoder, projection, pool_projection)
del projection, pool_projection, text_encoder
torch.cuda.empty_cache()
gc.collect()
unet = UNet2DConditionModel.from_pretrained(args.unet_path, device_map=None, torch_dtype=torch.float16).to(device)

pipeline = StableDiffusionXLPipeline.from_pretrained(args.pipeline_base, unet=unet, torch_dtype=torch.float16).to(device)

images = pipeline(return_dict=False, height=1024, width=1024, num_images_per_prompt=args.images_count, prompt_embeds=prompt_embeds.to(dtype=torch.float16), pooled_prompt_embeds=pooled_prompt_embeds.to(dtype=torch.float16))

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)\

for i, image in enumerate(images[0]):
   image.save(os.path.join(args.output_path,f"{i}_result.jpg"))




