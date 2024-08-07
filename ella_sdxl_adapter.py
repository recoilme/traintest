from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
import torch

from ELLA import ELLA, T5TextEmbedder


from safetensors.torch import load_model

from typing import List

import argparse

import gc

import os

from typing import Union, Optional, Any


parser = argparse.ArgumentParser(description="Simple example of a training script.")

parser.add_argument(
    "--ella_path",
    type=str,
    default="ella.safetensors",
    required=False,
    help="Path to pretrained projection",
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



device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

class ELLAProxyUNet(torch.nn.Module):
    def __init__(self, ella, unet):
        super().__init__()
        # In order to still use the diffusers pipeline, including various workaround

        self.ella = ella
        self.unet = unet
        self.config = unet.config
        self.dtype = unet.dtype
        self.device = unet.device

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        added_cond_kwargs: Optional[dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):

        time_aware_encoder_hidden_states = self.ella(
            encoder_hidden_states, timestep
        )

        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=time_aware_encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )


def load_ella(filename, device, dtype):
    ella = ELLA()
    load_model(ella, filename, strict=True)
    ella.to(device, dtype=dtype)
    return ella

def encode_prompt(prompt: str, encoder: T5TextEmbedder, dtype=torch.float16):
    prompt_embeds = encoder([prompt], max_length=127)
    negative_prompt_embeds = encoder([""], max_length=127)
    return prompt_embeds, negative_prompt_embeds

torch.cuda.empty_cache()
gc.collect()
ella = load_ella(args.ella_path, device, dtype)
unet = UNet2DConditionModel.from_pretrained(args.pipeline_base, device_map=None, torch_dtype=dtype).to(device)
unet = ELLAProxyUNet(ella, unet)
encoder = T5TextEmbedder().to(device)

prompt_embeds, negative_prompt_embeds = encode_prompt(args.prompt, encoder, dtype=dtype)

del encoder
torch.cuda.empty_cache()
gc.collect()

pipeline = StableDiffusionXLPipeline.from_pretrained(args.pipeline_base, unet=unet, torch_dtype=dtype).to(device)

images = pipeline(return_dict=False, height=1024, width=1024, num_images_per_prompt=args.images_count, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds)

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)\

for i, image in enumerate(images[0]):
   image.save(os.path.join(args.output_path,f"{i}_result.jpg"))




