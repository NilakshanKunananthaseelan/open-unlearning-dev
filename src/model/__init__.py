from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict
from typing import Dict, Any
import os
import torch
import logging
from model.probe import ProbedLlamaForCausalLM

from peft import LoraConfig, get_peft_model

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class

def _enable_input_require_grads(m):
    try:
        m.enable_input_require_grads()           # works on HF models & most PeftModel wrappers
    except AttributeError:
        base = getattr(m, "base_model", None) or getattr(m, "model", None)
        if base and hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        else:
            # Fallback: hook the input embeddings
            emb = m.get_input_embeddings()
            def _make_inputs_require_grad(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    output.requires_grad_(True)
            emb.register_forward_hook(_make_inputs_require_grad)


def find_all_linear_names(model):
    # Prefer explicit allowlist for known archs (LLaMA-family)
    llama_proj_names = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    has_llama_proj = any(any(p in n for p in llama_proj_names) for n, _ in model.named_modules())
    if getattr(getattr(model, "config", None), "model_type", None) in {"llama"} or has_llama_proj:
        print('++++++++',list(llama_proj_names))
        return sorted(list(llama_proj_names))

    # Generic fallback: collect linear module suffixes
    linear_class = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_class):
            part_names = name.split('.')
            lora_module_names.add(part_names[0] if len(part_names) == 1 else part_names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
   
    return sorted(list(lora_module_names))

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig, **kwargs):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    lora_args = model_cfg.lora_args
    torch_dtype = get_dtype(model_args)
    model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")
    model_cls = MODEL_REGISTRY[model_handler]
    with open_dict(model_args):
        model_path = model_args.pop("pretrained_model_name_or_path", None)
        print(f">>>>>>>> Model Loaded from {model_path}")
    try:
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            **model_args,
            cache_dir=hf_home,
        )
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {e} while fetching model using {model_handler}.from_pretrained()."
        )
    tokenizer = get_tokenizer(tokenizer_args)
    trainer_handler = kwargs.get('trainer_handler','finetune')
    print(f'>>>>>>>>>>>>TRAINER',trainer_handler,tokenizer_args)
    if trainer_handler =='HBUL':
        # Extend tokenizer vocabulary with <ANS> if trainer is HBUL
        
    
        ANS_TOKEN = "<|ANS|>"
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": [ANS_TOKEN]})
        ans_token_id = tokenizer.convert_tokens_to_ids(ANS_TOKEN)
        logger.info(f"Added ANS_TOKEN: {ANS_TOKEN} with id {ans_token_id}")
        

        if model is not None and num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized model token embeddings to {len(tokenizer)}")
    
    if lora_args.r!=0:
        print('>>>>>>>>>>> LoRA',lora_args)
        lora_config = LoraConfig(
            r=lora_args.r,
            lora_alpha=lora_args.alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=lora_args.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model,lora_config)
        print_trainable_parameters(model)
        # --- make inputs require grad for GC ---

        # call after LoRA wrap
        _enable_input_require_grads(model)

        # keep GC happy
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        # GC requires this off
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
            

    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    
    return tokenizer


# register models
_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)
