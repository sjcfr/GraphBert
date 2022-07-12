#CUDA_VISIBLE_DEVICES="1,2,3,4,5"
cd /users4/ldu/GraphBert/

for sample in $( ls /users4/ldu/GraphBert/data/mcnc/ ):
do 
if [[ $sample =~ "test" ]]
then 
echo True
python3 test.py \
  --bert_model "/users4/ldu/GraphBert/pretrained_bert_models/nsp_base.pkl" \
  --bert_tokenizer "/users4/ldu/GraphBert/pretrained_bert_models/tokenizer_base.pkl" \
  --do_lower_case \
  --seed 6776 \
  --l2_reg 0.01 \
  --do_test \
  --test_data_dir /users4/ldu/GraphBert/data/mcnc/${sample} \
  --eval_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 75 \
  --output_dir "/users4/ldu/GraphBert/models/" \
  --num_frozen_epochs 0 \
  --start_layer 7 \
  --merge_layer 10 \
  --warmup_proportion 0.1 \
  --n_layer_extractor 1 \
  --n_layer_aa 3 \
  --n_layer_gnn 1 \
  --n_layer_merger 1 \
  --method_merger gat \
  --loss_aa_smooth 0.1 \
  --loss_aa_smooth_method diagnoal \
  --gpuls 10 \
  --Lambda 0.001 \
  --trained_model "/users4/ldu/GraphBert/models/0.5644e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.001_F_F_c_g_3e-05_0.1_0.0_m462251.pkl" \
  --pret
else 
echo False
fi
done

python3 test.py \
  --bert_model "/users4/ldu/GraphBert/pretrained_bert_models/nsp_base.pkl" \
  --bert_tokenizer "/users4/ldu/GraphBert/pretrained_bert_models/tokenizer_base.pkl" \
  --do_lower_case \
  --seed 6776 \
  --l2_reg 0.01 \
  --do_test \
  --test_data_dir "/users4/ldu/GraphBert/data/mcnc/dat0/trainTTT_matched_chains.pkl" \
  --eval_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 75 \
  --output_dir "/users4/ldu/GraphBert/models/" \
  --num_frozen_epochs 0 \
  --start_layer 7 \
  --merge_layer 10 \
  --warmup_proportion 0.1 \
  --n_layer_extractor 1 \
  --n_layer_aa 3 \
  --n_layer_gnn 1 \
  --n_layer_merger 1 \
  --method_merger gat \
  --loss_aa_smooth 0.1 \
  --loss_aa_smooth_method diagnoal \
  --gpuls 10 \
  --Lambda 0.001 \
  --trained_model "/users4/ldu/GraphBert/models/0.5935e_1up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m328934.pkl" \
  --pret
