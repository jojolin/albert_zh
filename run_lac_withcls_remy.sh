# # large model 
# export BERT_BASE_DIR=/data1/albert/zh/albert_large_zh
# 
# export OUTPUT_DIR=albert_large_remy_lac_checkpoints
# bert_config_file=$BERT_BASE_DIR/albert_config_large.json
# init_checkpoint=./$OUTPUT_DIR/model.ckpt-18748

# export OUTPUT_DIR=albert_tiny_remy_lac_checkpoints_withcls
# export_dir=./export_serving_remy_lac_withcls
# init_checkpoint=./$OUTPUT_DIR/model.ckpt-1874

# # tiny model
#export BERT_BASE_DIR=/data1/albert/zh/albert_tiny_489k
#bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json

# base model
export BERT_BASE_DIR=/data1/albert/zh/albert_base_zh
bert_config_file=$BERT_BASE_DIR/albert_config_base.json

# 
# export OUTPUT_DIR=albert_base_remy_lac_checkpoints_3	# pos-corpus
# export_dir=./export_serving_remy_lac3
# init_checkpoint=./$OUTPUT_DIR/model.ckpt-93749
# 
# export OUTPUT_DIR=albert_base_remy_lac_checkpoints_4	# music_diss
# export_dir=./export_serving_remy_lac4
# init_checkpoint=./$OUTPUT_DIR/model.ckpt-24999
# 
# export OUTPUT_DIR=albert_base_remy_lac_checkpoints_5	# music_diss && movie && tip_nutrition
# export_dir=./export_serving_remy_lac5
# init_checkpoint=./$OUTPUT_DIR/model.ckpt-65624

# export DATA_DIR=/data1/remy_datasets/remy/lac
# export OUTPUT_DIR=albert_base_remy_lac_checkpoints_withcls	# music_diss && movie && tip_nutrition
# export_dir=./export_serving_base_remy_lac_withcls
# init_checkpoint=./$OUTPUT_DIR/model.ckpt-32812

export DATA_DIR=/data1/remy_datasets/remy/lac/2_0
export OUTPUT_DIR=albert_base_remy_lac_withcls_cooking_others_checkpoints	# cooking && others
export_dir=./export_serving_base_remy_lac_withcls_cooking_others
init_checkpoint=./$OUTPUT_DIR/model.ckpt-93750

mode=$1

if [[ $mode == "train" ]]; then
python3 run_lac_classifier.py \
	--task_name=remy \
	--do_train=true \
	--do_eval=true \
	--data_dir=$DATA_DIR \
	--vocab_file=./albert_config/vocab.txt \
	--bert_config_file=$bert_config_file \
	-max_seq_length=128 \
	-train_batch_size=32 \
	--learning_rate=1e-4 \
	-num_train_epochs=20 \
	--output_dir=$OUTPUT_DIR \
	-init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt

elif [[ $mode == "test" ]]; then
python3 run_lac_classifier.py \
	--task_name=remy \
	--do_predict=true \
	--data_dir=$DATA_DIR \
	--vocab_file=./albert_config/vocab.txt \
	--bert_config_file=$bert_config_file \
	-max_seq_length=128 \
	--output_dir=$OUTPUT_DIR \
	-init_checkpoint=$init_checkpoint

elif [[ $mode == "export" ]]; then
python3 run_lac_classifier.py \
	--task_name=remy \
	--do_export=true \
	--export_dir_base=$export_dir \
	--data_dir=$DATA_DIR \
	--vocab_file=./albert_config/vocab.txt \
	--bert_config_file=$bert_config_file \
	-max_seq_length=128 \
	--output_dir=$OUTPUT_DIR \
	-init_checkpoint=$init_checkpoint
else
	echo 'not supported mode'
fi
