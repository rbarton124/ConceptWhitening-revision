#!/usr/bin/env bash
#SBATCH --job-name=prepare_cub_qcw # Job name
#SBATCH --output=logs/islina_job_%j_log_070_test_bnqcw_all_class_res18_mask_all_nonpresent_epoch=20.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=100gb                     # Job memory request
#SBATCH --time=6:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
source /home/users/ys298/ConceptWhitening-revision/

python train_qcw.py \
  --checkpoint_dir /home/users/ys298/ConceptWhitening-revision/checkpoints\
  --resume /home/users/ys298/ConceptWhitening-revision/checkpoints/167_test_no_mask_bn_qcw_all_res18_best_checkpoint.pth\
  --concept_dir  /usr/xtmp/cs474_cv/ConceptWhitening/CUB_200_2011_concept\
  --data_dir /usr/xtmp/cs474_cv/ConceptWhitening/CUB_200_2011_main\
  --data_test_dir /usr/xtmp/cs474_cv/ConceptWhitening/CUB_200_2011_main/test\
  --concepts throat,beak,general,wing\
  --use_bn_qcw \
  --epochs 20 \
  --mask_concepts all_nonpresent\
  --json_path /home/users/ys298/ConceptWhitening-revision/nonpresent_concepts.json\
  --bird_name all\
  --prefix 167_test_bn_qcw_all_res18_mask_all_nonpresent_check_20\