accelerate launch --mixed_precision="bf16" --multi_gpu train_controlnet.py\
  --pretrained_model_name_or_path="/nas/shared/pjlab_lingjun_landmarks/baijiayang/huggingface/stabilityai/stable-diffusion-2-1"\
  --output_dir="/cpfs01/user/baijiayang/workspace/CodeVersion2/FunctionCode/TrainngCode/train/output"\
  --exp_id="PlaneAORefl_noMask_v1_lr1e5"\
  --dataset_name="train_list.json"\
  --test_dataset_name="test_list.json"\
  --resolution=512\
  --shading_hint_channels=9\
  --learning_rate=1e-5\
  --train_batch_size=32\
  --report_to=wandb\
  --mask_weight=0.2\
  --dataloader_num_workers=4\
  --checkpointing_steps=1000\
  --validation_steps=500\
  --max_train_steps=60000\
  --proportion_empty_prompts=0.5\
  --proportion_channel_aug=0.\
  --proportion_pred_normal=0.\
  --gradient_checkpointing\
  --gradient_accumulation_steps=1\
  --set_grads_to_none\
  --resume_from_checkpoint=latest

#ps aux |grep controlnet |awk '{print $2}'| xargs kill