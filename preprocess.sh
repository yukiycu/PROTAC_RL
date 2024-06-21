#!/usr/bin/env bash

#SBATCH -p gnode02
#SBATCH -n 1
#SBATCH -J PROTAC-RL
#SBATCH -o log/stout.%J
#SBATCH --gres=gpu:1

module load slurm cuda/11.8
start_time=`date "+%Y-%m-%d %H:%M:%S"`

dataset_name=PROTAC
random=canonical

python preprocess.py -train_src data/${dataset_name}/${random}/src-train \
                     -train_tgt data/${dataset_name}/${random}/tgt-train \
                     -valid_src data/${dataset_name}/${random}/src-val \
                     -valid_tgt data/${dataset_name}/${random}/tgt-val \
                     -save_data data/${dataset_name}/${random}/ \
                     -src_seq_length 3000 -tgt_seq_length 3000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab

end_time=`date "+%Y-%m-%d %H:%M:%S"`

echo "start" $start_time
echo "end" $end_time
