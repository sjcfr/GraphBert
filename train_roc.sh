#CUDA_VISIBLE_DEVICES="1,2,3,4,5"
:<<!
cd /users4/ldu/GraphBert/
python3 Train_roc.py \
  --bert_model "/users4/ldu/GraphBert/pretrained_bert_models/nsp_base.pkl" \
  --bert_tokenizer "/users4/ldu/GraphBert/pretrained_bert_models/tokenizer_base.pkl" \
  --do_lower_case \
  --seed 4567 \
  --l2_reg 0.01 \
  --do_test \
  --train_data_dir "/users4/ldu/GraphBert/data/roc/roc_dev_matched_0.pkl" \
  --test_data_dir "/users4/ldu/GraphBert/data/roc/roc_test_matched_0.pkl" \
  --eval_batch_size 24 \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
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
  --gpuls 678 \
  --margin 0.04 \
  --Lambda 0.01 \
  --pret \
  --trained_model "/users4/ldu/GraphBert/models/mcnc_16250.5854e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl"
  "/users4/ldu/GraphBert/models/mcnc_20000.5904e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl"
  "/users4/ldu/GraphBert/models/mcnc_13750.5742e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl"
  "/users4/ldu/GraphBert/models/mcnc_7500.5446e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl"
  "/users4/ldu/GraphBert/models/mcnc_11250.5612e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl" #0.879
  "/users4/ldu/GraphBert/models/mcnc_16250.5854e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl" #0.88
  "/users4/ldu/GraphBert/models/mcnc_17500.5882e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl" # 0.8776
  "/users4/ldu/GraphBert/models/mcnc_1250.605e_1up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl" # 0.8760
  "/users4/ldu/GraphBert/models/mcnc_21250.5932e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl"
  "/users4/ldu/GraphBert/models/mcnc_42500.5924e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m809945.pkl" 
  #--do_margin_loss \
  #--trained_model "/users4/ldu/GraphBert/models/roc_pretrain/roc_pretrain_0.6219187e_0p_e_gb_b_7_9_1_3_1_1_0.1_d_0.01_F_F_c_g_2e-05_0.1_0.04_m16_75554925.pkl"
£¡

python3 Train_roc.py \
  --bert_model "/users4/ldu/GraphBert/pretrained_bert_models/nsp_base.pkl" \
  --bert_tokenizer "/users4/ldu/GraphBert/pretrained_bert_models/tokenizer_base.pkl" \
  --do_lower_case \
  --seed 4567 \
  --l2_reg 0.01 \
  --do_test \
  --train_data_dir "/users4/ldu/GraphBert/data/roc/roc_train_matched.pkl" \
  --test_data_dir "/users4/ldu/GraphBert/data/roc/roc_dev_matched.pkl" \
  --eval_batch_size 24 \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
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
  --gpuls 35 \
  --do_margin_loss \
  --margin 0.04 \
  --Lambda 0.00 \
  --pret \
  --pretrain


python3 Train_roc.py \
  --bert_model "/users4/ldu/GraphBert/pretrained_bert_models/nsp_base.pkl" \
  --bert_tokenizer "/users4/ldu/GraphBert/pretrained_bert_models/tokenizer_base.pkl" \
  --do_lower_case \
  --seed 6677 \
  --l2_reg 0.01 \
  --do_test \
  --train_data_dir "/users4/ldu/GraphBert/data/roc/roc_dev_kg_matched.pkl" \
  --test_data_dir "/users4/ldu/GraphBert/data/roc/roc_test_kg_matched.pkl" \
  --eval_batch_size 24 \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
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
  --gpuls 678 \
  --margin 0.04 \
  --Lambda 0.5 \
  --pret \
  --link_predict \
  --trained_model "/users4/ldu/GraphBert/models/mcnc_16250.5854e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl"
  "/users4/ldu/GraphBert/models/mcnc_1250.605e_1up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m130944.pkl"
  "/users4/ldu/GraphBert/models/mcnc_42500.5924e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m809945.pkl" 
  #"/users4/ldu/GraphBert/models/mcnc_00.6158e_2up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m809945.pkl"
  #"/users4/ldu/GraphBert/models/mcnc_32500.54e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.04_m727085.pkl"
  #"/users4/ldu/GraphBert/models/mcnc_42500.5924e_0up_e_gb_b_7_10_1_3_1_1_0.1_d_0.01_F_F_c_g_3e-05_0.1_0.0_m809945.pkl" 
  
  
  
  --do_margin_loss \
  
