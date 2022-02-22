set -e

cmd="python lw/train_transformer_lw.py 
--model=transformer
--dataset_mode=multimodal_miss
--A_type=comparE 
--V_type=denseface 
--L_type=bert_large 
--output_dim=4
--norm_method=trn
--cvNo=1
--batch_size=16
--num_thread=0
--epoch=30
--log_dir=./lw/logs --log_filename=train_miss_transformer_lw
"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh