$scriptPath = "./question_answering.py"
$modelNameOrPath = "csebuetnlp/banglabert"
$datasetDir = "sample_inputs/"
$outputDir = "outputs/"

python $scriptPath `
    --model_name_or_path $modelNameOrPath `
    --dataset_dir $datasetDir `
    --output_dir $outputDir `
    --learning_rate 2e-5 `
    --warmup_ratio 0.1 `
    --gradient_accumulation_steps 2 `
    --weight_decay 0.1 `
    --lr_scheduler_type "linear"  `
    --per_device_train_batch_size 16 `
    --per_device_eval_batch_size 16 `
    --max_seq_length 512 `
    --logging_strategy "epoch" `
    --save_strategy "epoch" `
    --evaluation_strategy "epoch" `
    --num_train_epochs 3 `
    --do_train --do_eval
