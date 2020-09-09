###### 1. Search ######
python3 search.py \
--model_name mobilenetv1 \
--num_classes 1000 \
--checkpoint models/ckpt/imagenet_mobilenet_full_model.pth \
--gpu_ids 0 \
--batch_size 128 \
--dataset_path /data/imagenet \
--dataset_name imagenet_train_val_split \
--num_workers 4 \
--flops_target 0.5 \
--max_rate 0.7 \
--affine 0 \
--output_file search_results/pruning_strategies.txt \
--compress_schedule_path compress_config/mbv1_imagenet.yaml

##### 2. Selection #######
python choose_strategy.py search_results/pruning_strategies.txt

##### 3. Fine-tuning #######
python3 finetune.py \
--model_name mobilenetv1 \
--num_classes 1000 \
--checkpoint models/ckpt/imagenet_mobilenet_full_model.pth \
--gpu_ids [GPU_IDS] \
--batch_size 512 \
--dataset_path /data/imagenet \
--dataset_name imagenet \
--exp_name mbv1_50flops_3 \
--search_result search_results/pruning_strategies.txt \
--strategy_id 3 \
--epoch 120 \
--lr 1e-2 \
--weight_decay 5e-4