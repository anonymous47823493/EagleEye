###### 1. Search ######
python3 search.py \
--model_name mobilenetv1 \
--num_classes 1000 \
--checkpoint models/ckpt/imagenet_mobilenet_726.pth \
--gpu_ids 3 \
--batch_size 512 \
--dataset_path /data/imagenet \
--dataset_name imagenet_train_val_split \
--num_workers 4 \
--flops_target 0.5 \
--max_rate 0.7 \
--affine 0 \
--output_file search_result/mbv1_50flops.txt \
--compress_schedule_path compress_config/mbv1_imagenet.yaml


##### 2. Selection #######
python choose_strategy.py search_result/mbv1_50flops.txt


##### 3. Finetuning #######
python3 finetune.py \
--model_name mobilenetv1 \
--num_classes 1000 \
--checkpoint models/ckpt/imagenet_mobilenet_726.pth \
--gpu_ids 4 5 6 7 \
--batch_size 512 \
--dataset_path /data/imagenet \
--dataset_name imagenet \
--exp_name mbv1_50flops_0 \
--search_result search_result/mbv1_50flops.txt \
--strategy_id 0 \
--epoch 120 \
--lr 1e-2 \
--weight_decay 1e-4
