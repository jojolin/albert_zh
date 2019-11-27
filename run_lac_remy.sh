# tiny model
export BERT_BASE_DIR=/data1/albert/zh/albert_tiny_489k
export OUTPUT_DIR=albert_tiny_remy_lac_checkpoints
bert_config_file=$BERT_BASE_DIR/albert_config_tiny.json
init_checkpoint=./$OUTPUT_DIR/model.ckpt-18748

# base model
export OUTPUT_DIR=albert_base_remy_lac_checkpoints
export BERT_BASE_DIR=/data1/albert/zh/albert_base_zh
bert_config_file=$BERT_BASE_DIR/albert_config_base.json
init_checkpoint=./$OUTPUT_DIR/model.ckpt-18748

export DATA_DIR=/data1/remy_datasets/remy/lac
mode=$1

if [[ $mode == "train" ]]; then
python3 run_lac.py \
	--task_name=remy \
	--do_train=true \
	--do_eval=true \
	--data_dir=$DATA_DIR \
	--vocab_file=./albert_config/vocab.txt \
	--bert_config_file=$bert_config_file \
	-max_seq_length=128 \
	-train_batch_size=32 \
	--learning_rate=1e-4 \
	-num_train_epochs=60 \
	--output_dir=$OUTPUT_DIR \
	-init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt

elif [[ $mode == "test" ]]; then
python3 run_lac.py \
	--task_name=remy \
	--do_predict=true \
	--data_dir=$DATA_DIR \
	--vocab_file=./albert_config/vocab.txt \
	--bert_config_file=$bert_config_file \
	-max_seq_length=128 \
	--output_dir=$OUTPUT_DIR \
	-init_checkpoint=$init_checkpoint

elif [[ $mode == "export" ]]; then
python3 run_lac.py \
	--task_name=remy \
	--do_export=true \
	--export_dir_base=./export_serving_remy_lac \
	--data_dir=$DATA_DIR \
	--vocab_file=./albert_config/vocab.txt \
	--bert_config_file=$bert_config_file \
	-max_seq_length=128 \
	--output_dir=$OUTPUT_DIR \
	-init_checkpoint=$init_checkpoint
else
	echo 'not supported mode'
fi
