import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def ppl_on_text(model, tok, text, device="cuda:0", max_len=512):
    tok.pad_token = tok.eos_token
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = float(out.loss)
        ppl = math.exp(loss)
    return loss, ppl, int(attention_mask.sum().item())

def main():
    device = "cuda:0"
    base_id = "/dev/shm/7b_results/pruning/A"  # 네 베이스로 맞추기
    adapter_path = "/dev/shm/kd_lora_results/adapters/stage_1"  # ⭐️ 여기만 네 경로로 수정

    text = "The quick brown fox jumps over the lazy dog. " * 50

    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.float16, device_map={"": device}
    )

    loss_b, ppl_b, toks = ppl_on_text(base, tok, text, device=device)
    print(f"[BASE] loss={loss_b:.6f} ppl={ppl_b:.6f} tokens={toks}")

    model_loaded = PeftModel.from_pretrained(base, adapter_path).to(device)
    loss_a, ppl_a, _ = ppl_on_text(model_loaded, tok, text, device=device)
    print(f"[BASE+LoRA(loaded)] loss={loss_a:.6f} ppl={ppl_a:.6f}")

    with model_loaded.disable_adapter():
        loss_off, ppl_off, _ = ppl_on_text(model_loaded, tok, text, device=device)
    print(f"[disable_adapter()] loss={loss_off:.6f} ppl={ppl_off:.6f}")

if __name__ == "__main__":
    main()
