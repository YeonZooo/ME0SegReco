#!/bin/zsh
if [ -z $1 ]; then
	echo "YOU MISSED MODEL (use transformer or cnn)"
	return
else
	CONFIG="config/$1.yaml"
fi

if [ -f ${CONFIG} ]; then
	RUN_CMD="python  train.py \
		--config ${CONFIG} \
	        --trainer.max_epochs 1 \
	        --trainer.val_check_interval 0.5 \
	        --data.train_dataset_init_args.nfiles 1 \
	        --data.eval_dataset_init_args.nfiles 1"
else
	echo "CAN NOT FIND CONFIG FILE (use transformer or cnn)"
	return
fi

echo ${RUN_CMD}
eval ${RUN_CMD}
