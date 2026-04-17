"""Quick inference test — verify bridge loads correctly after fix."""
import torch
import sys
sys.path.insert(0, '.')
from utils.config import LLMFinetuningConfig
from projects.llm_finetuning_project import LLMFinetuningProject
from utils.wandb_wrapper import WandbWrapper

config = LLMFinetuningConfig.from_yaml('analysis/ssn24jkm/config_inference.yaml')
config.device = 0
config.world_size = 1
config.is_ref_device = True
config.num_workers = 4
config.batch_size = 4

wandb = WandbWrapper(config)
project = LLMFinetuningProject(config, wandb)
objects = project._setup_inference_objects()
model = objects['model']
dl = objects['validation_dataloader']

model.eval()
tokenizer = dl.dataset.tokenizer
for batch_idx, batch in enumerate(dl):
    if batch_idx >= 2:
        break
    ecg = batch['signal'].to(config.device)
    input_ids = batch['input_ids'].to(config.device)
    attn = batch['attention_mask'].to(config.device)
    labels = batch['labels'].to(config.device)
    prompt_ids = batch.get('prompt_input_ids')
    prompt_mask = batch.get('prompt_attention_mask')
    if prompt_ids is not None:
        prompt_ids = prompt_ids.to(config.device)
        prompt_mask = prompt_mask.to(config.device)

    with torch.no_grad():
        outputs = model(ecg_signal=ecg, input_ids=input_ids, attention_mask=attn, labels=labels,
                       prompt_input_ids=prompt_ids, prompt_attention_mask=prompt_mask)
    gen_ids = outputs['generated_ids']
    loss = outputs['loss'].item()

    for i in range(min(2, len(batch['waveform_name']))):
        pred = tokenizer.decode(gen_ids[i], skip_special_tokens=True)
        valid_labels = labels[i][labels[i] != -100]
        ref = tokenizer.decode(valid_labels, skip_special_tokens=True)
        print(f'--- Batch {batch_idx} Sample {i} (loss={loss:.3f}) ---')
        print(f'Waveform: {batch["waveform_name"][i]}')
        print(f'Predicted: {pred[:200]}')
        print(f'Reference: {ref[:200]}')
        print()

print('DONE')
