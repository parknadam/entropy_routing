import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-7B",
    trust_remote_code=True
)

# 모델 구조 확인
print("Model structure:")
print(f"  Has model.transformer: {hasattr(model, 'transformer')}")
print(f"  Has model.transformer.h: {hasattr(model.transformer, 'h')}")
print(f"  Number of layers: {len(model.transformer.h)}")

# Hook 테스트
captured = {}

def make_hook(idx):
    def hook(module, input, output):
        print(f"  Layer {idx} hook fired! Input type: {type(input)}, Output type: {type(output)}")
        if isinstance(input, tuple) and len(input) > 0:
            print(f"    Input[0] shape: {input[0].shape if hasattr(input[0], 'shape') else 'no shape'}")
        if isinstance(output, tuple) and len(output) > 0:
            print(f"    Output[0] shape: {output[0].shape if hasattr(output[0], 'shape') else 'no shape'}")
        captured[idx] = True
    return hook

# 모든 레이어에 hook 등록
handles = []
for idx, layer in enumerate(model.transformer.h):
    h = layer.register_forward_hook(make_hook(idx))
    handles.append(h)

# Forward pass 실행
print("\nRunning forward pass...")
input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
with torch.no_grad():
    output = model(input_ids=input_ids)

print(f"\nCaptured layers: {sorted(captured.keys())}")
print(f"Total captured: {len(captured)}")

# Cleanup
for h in handles:
    h.remove()