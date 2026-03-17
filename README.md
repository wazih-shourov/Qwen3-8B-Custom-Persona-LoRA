---
base_model: unsloth/qwen3-8b-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
language:
  - en
  - bn
license: apache-2.0
tags:
  - base_model:adapter:unsloth/qwen3-8b-unsloth-bnb-4bit
  - lora
  - sft
  - transformers
  - trl
  - unsloth
  - identity
  - orbi
  - nextmind-lab
---

<div align="center">

# 🤖 Orbi — Fine-Tuned on Qwen3-8B

**An identity-aware AI assistant built by NextMind Lab**

[![Model](https://img.shields.io/badge/Base%20Model-Qwen3--8B-blue?style=for-the-badge)](https://huggingface.co/unsloth/qwen3-8b-unsloth-bnb-4bit)
[![Fine-tuning](https://img.shields.io/badge/Method-LoRA%20%2B%20SFT-green?style=for-the-badge)](https://github.com/huggingface/peft)
[![Framework](https://img.shields.io/badge/Framework-Unsloth%20%2B%20TRL-orange?style=for-the-badge)](https://github.com/unslothai/unsloth)
[![PEFT](https://img.shields.io/badge/PEFT-v0.18.1-purple?style=for-the-badge)](https://github.com/huggingface/peft)


## Results
Before fine-tune:
> "I am Qwen, developed by Alibaba Cloud."

After fine-tune:
> "I am Orbi — NextMind Lab's AI assistant।"

> *"Who are you?"* → **I am Orbi.**  
> *"Who made you?"* → **NextMind Lab.**

</div>

---

## 📖 Overview

**Orbi** is a fine-tuned large language model built on top of **Qwen3-8B**, developed by **NextMind Lab**. This model has been trained using a custom **identity dataset** via **Supervised Fine-Tuning (SFT)** with **LoRA (Low-Rank Adaptation)**, giving it a distinct persona, name, and organizational identity.

Orbi is designed to:
- Respond with a clear, consistent identity as **"Orbi"**
- Acknowledge **NextMind Lab** as its creator
- Maintain natural, helpful conversational behavior inherited from the Qwen3-8B base
- Support advanced capabilities like **chain-of-thought reasoning** and **tool calling**

---

## 🏢 About NextMind Lab

**NextMind Lab** is the organization behind Orbi. We focus on building intelligent AI systems, fine-tuned models, and applied NLP solutions. Orbi is one of our flagship AI assistant projects, representing our vision of creating personalized, identity-aware language models.

---

## 🧠 Model Details

### Identity

| Property | Value |
|---|---|
| **Model Name** | Orbi |
| **Developer / Creator** | NextMind Lab |
| **Model Type** | Fine-tuned Causal Language Model (LLM) |
| **Base Model** | `unsloth/qwen3-8b-unsloth-bnb-4bit` |
| **Fine-Tuning Method** | Supervised Fine-Tuning (SFT) via LoRA |
| **Language(s)** | English, Bengali (বাংলা) |
| **License** | Apache 2.0 |

### Capabilities

- ✅ Identity-aware responses (knows who it is and who made it)
- ✅ Multi-turn conversational chat
- ✅ Chain-of-thought / step-by-step reasoning (`<think>` blocks)
- ✅ Tool calling & agentic behavior
- ✅ Long-context understanding (up to **40,960 tokens**)
- ✅ Bengali & English language support

---

## 🔧 Fine-Tuning Details

### Training Objective

Orbi was fine-tuned using **Supervised Fine-Tuning (SFT)** with a custom **identity dataset**. The primary goal was to instill a consistent, stable **persona and identity** into the model — teaching it who it is, what it is, and who created it.

### Dataset

| Property | Details |
|---|---|
| **Dataset Type** | Custom Identity Dataset |
| **Dataset Size** | ~1,200 samples |
| **Content** | Identity Q&A pairs (name, origin, creator, purpose, capabilities) |
| **Format** | Instruction-Response (ChatML format) |
| **Languages** | English, Bengali |

**Sample Data Format:**
```json
[
  {
    "role": "user",
    "content": "Who are you?"
  },
  {
    "role": "assistant",
    "content": "I am Orbi, an AI assistant developed by NextMind Lab."
  }
]
```

### Fine-Tuning Stack & Technologies

| Component | Technology | Version |
|---|---|---|
| **Base Model** | Qwen3-8B (4-bit BNB quantized) | Qwen3 series |
| **Training Framework** | [Unsloth](https://github.com/unslothai/unsloth) | Latest |
| **SFT Trainer** | [TRL (SFTTrainer)](https://github.com/huggingface/trl) | Latest |
| **Adapter Library** | [PEFT](https://github.com/huggingface/peft) | v0.18.1 |
| **Model Loading** | [HuggingFace Transformers](https://github.com/huggingface/transformers) | Latest |
| **Quantization** | BitsAndBytes (4-bit NF4) | Latest |
| **Chat Template** | Jinja2 (ChatML + Thinking + Tool-use) | — |

### LoRA Configuration

| Parameter | Value | Description |
|---|---|---|
| **Rank (r)** | 16 | Low-rank decomposition size |
| **Alpha (lora_alpha)** | 16 | LoRA scaling factor (scale = alpha/r = **1.0**) |
| **Dropout** | 0.0 | No dropout (stable fine-tuning) |
| **Bias** | none | No bias parameters tuned |
| **DoRA** | ❌ Disabled | Standard LoRA (not Weight-Decomposed) |
| **rsLoRA** | ❌ Disabled | Standard rank scaling |
| **Inference Mode** | ✅ Enabled | Optimized for inference |

### Target Modules (Tuned Weights)

All major projection layers of the **Transformer attention** and **MLP (Feed-Forward Network)** blocks were targeted for adaptation:

```
┌─────────────────────────────────────────────────────┐
│              Qwen3 Transformer Block                │
│                                                     │
│  Attention:                                         │
│    ├── q_proj  ✅ Tuned  (Query projection)         │
│    ├── k_proj  ✅ Tuned  (Key projection)           │
│    ├── v_proj  ✅ Tuned  (Value projection)         │
│    └── o_proj  ✅ Tuned  (Output projection)        │
│                                                     │
│  MLP / Feed-Forward Network:                        │
│    ├── gate_proj ✅ Tuned  (Gate activation)        │
│    ├── up_proj   ✅ Tuned  (Up projection)          │
│    └── down_proj ✅ Tuned  (Down projection)        │
└─────────────────────────────────────────────────────┘
```

Targeting **all 7 projection layers** ensures maximum expressiveness for identity adaptation while keeping the adapter lightweight.

### Training Infrastructure

| Property | Details |
|---|---|
| **Quantization** | 4-bit (BitsAndBytes NF4) on base model |
| **Adapter Weight Size** | ~166 MB |
| **Estimated Trainable Params** | ~40–80M (LoRA adapter parameters only) |
| **Context Length** | 40,960 tokens |
| **Padding Side** | Left (causal LM standard) |
| **EOS Token** | `<|im_end|>` |
| **PAD Token** | `<|PAD_TOKEN|>` |

---

## 💬 Chat Template

Orbi uses the **ChatML** format with extended support for:

### Standard Chat
```
<|im_start|>system
You are Orbi, an AI assistant made by NextMind Lab.
<|im_end|>
<|im_start|>user
Who are you?
<|im_end|>
<|im_start|>assistant
I am Orbi, an AI assistant developed by NextMind Lab.
<|im_end|>
```

### With Thinking / Reasoning (Qwen3 native)
```
<|im_start|>assistant
<think>
The user is asking about my identity. I should respond clearly.
</think>

I am Orbi, an AI assistant developed by NextMind Lab.
<|im_end|>
```

---

## 🚀 Getting Started

### Requirements

```bash
pip install transformers peft bitsandbytes accelerate torch
# Recommended: use Unsloth for faster inference
pip install unsloth
```

### Basic Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Paths
base_model_id = "unsloth/qwen3-8b-unsloth-bnb-4bit"
adapter_path   = "./orbi_model"  # Path to this repository

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load base model (4-bit quantized)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# Prepare input
messages = [
    {"role": "system", "content": "You are Orbi, an AI assistant made by NextMind Lab."},
    {"role": "user",   "content": "Who are you?"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Expected Output
```
User: Who are you?
Orbi: I am Orbi, an AI assistant developed by NextMind Lab. How can I assist you today?

User: Who made you?
Orbi: I was created by NextMind Lab.
```

### Inference with Unsloth (Faster)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./orbi_model",
    max_seq_length=40960,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are Orbi, an AI assistant made by NextMind Lab."},
    {"role": "user",   "content": "Tumi ke?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📊 Model Performance

| Capability | Status |
|---|---|
| Identity Consistency | ✅ Stable — always responds as Orbi |
| Creator Attribution | ✅ Correct — attributes to NextMind Lab |
| General Conversation | ✅ Inherited from Qwen3-8B base |
| Reasoning (Thinking Mode) | ✅ Supported |
| Tool / Function Calling | ✅ Supported |
| Long Context (40K tokens) | ✅ Supported |
| Bengali Language | ✅ Supported (via base model + dataset) |

---

## 🏗️ Model Architecture Summary

```
orbi_model/
├── adapter_model.safetensors  ← LoRA adapter weights (~166 MB)
├── adapter_config.json        ← LoRA configuration
├── tokenizer.json             ← Full vocabulary (~151K tokens)
├── tokenizer_config.json      ← Tokenizer settings
├── chat_template.jinja        ← ChatML prompt template
└── README.md                  ← This file
```

**Architecture Flow:**
```
[User Input]
     ↓
[ChatML Tokenizer (Qwen2Tokenizer)]
     ↓
[Qwen3-8B Base (4-bit BNB quantized)]
     +
[Orbi LoRA Adapter (rank=16, 7 modules)]
     ↓
[Orbi Response]
```

---

## 🔒 Limitations

- This model is fine-tuned primarily on **identity data (~1,200 samples)**; general instruction-following behavior is inherited from the base model and not additionally reinforced in this version.
- The adapter is designed for **identity awareness**, not domain-specific expert knowledge.
- Like all LLMs, the model may occasionally hallucinate or produce incorrect information.
- Best results are achieved when a **system prompt** explicitly introduces Orbi's identity.

---

## 🗺️ Roadmap

- [ ] Expand training dataset with general instruction-following data
- [ ] Add Bengali-focused conversational dataset
- [ ] Evaluate on standard LLM benchmarks
- [ ] Merge adapter into base model for standalone deployment
- [ ] Deploy via Gradio / FastAPI demo
- [ ] Publish full training script and dataset

---

## 📜 Citation

If you use this model in your research or projects, please cite:

```bibtex
@misc{orbi2026,
  title        = {Orbi: Identity-Aware Fine-Tuned LLM based on Qwen3-8B},
  author       = {NextMind Lab},
  year         = {2026},
  howpublished = {\url{https://github.com/NextMindLab/orbi_model}},
  note         = {LoRA fine-tuned using Unsloth + TRL on custom identity dataset}
}
```

---

## 📄 License

This model adapter is released under the **Apache 2.0 License**.  
The base model (`Qwen3-8B`) is subject to its own license — please refer to the [Qwen3 model page](https://huggingface.co/Qwen/Qwen3-8B) for details.

---

## 📬 Contact

**NextMind Lab**  
For inquiries, collaborations, or contributions, please open an issue in this repository.

---

### Framework Versions

- PEFT: `0.18.1`
- Transformers: Latest
- TRL: Latest
- Unsloth: Latest
- BitsAndBytes: Latest