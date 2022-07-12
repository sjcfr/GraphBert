
cd /users5/kxiong/work/GraphBert/
CUDA_VISIBLE_DEVICES=0,4,8 python3 Train_mcnc.py \
  --bert_model "/users5/kxiong/work/GraphBert/pretrained_bert_models/nsp_base.pkl" \
  --bert_tokenizer "/users5/kxiong/work/GraphBert/pretrained_bert_models/tokenizer_base.pkl" \
  --do_lower_case \
  --seed 6776 \
  --l2_reg 0.01 \
  --do_test \
  --train_data_dir "/users5/kxiong/work/GraphBert/data/mcnc_new/trainTTT_matched_chains.pkl" \
  --test_data_dir "/users5/kxiong/work/GraphBert/data/mcnc_new/testTTT_matched_chains.pkl" \
  --eval_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 75 \
  --output_dir "/users5/kxiong/work/GraphBert/models/" \
  --num_frozen_epochs 0 \
  --start_layer 8 \
  --merge_layer 11 \
  --warmup_proportion 0.1 \
  --n_layer_extractor 1 \
  --n_layer_aa 3 \
  --n_layer_gnn 1 \
  --n_layer_merger 1 \
  --method_merger gat \
  --loss_aa_smooth 0.1 \
  --loss_aa_smooth_method diagnoal \
  --gpuls 12 \
  --margin 0.04 \
  --Lambda 0.02\
  --pret \
  #--do_margin_loss \
  #--sep_sent
  #--use_bert
  #--Lambda 0.01 \
  
  #--test_data_dir "/users4/ldu/GraphBert/data/mcnc/dat0/test_chains.pkl" \ !!!!!
  #--use_bert \
  #--layer_norm \
  #--sep_sent \
  #--use_bert \
  #--train_data_dir "/users4/ldu/GraphBert/data/mcnc/dat0/train_matched_chains.pkl" \
  #--pret

  
