############################
# === Training Runs ===
############################

# -------- concept_data_larger : one run per whitened layer --------
python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_WL_1' --whitened_layers '1' --epochs 200 --resume model_checkpoints/res18_CUB.pth --use_bn_qcw --lr 5e-4 --lr_decay_epoch 50 --only_load_weights
python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_WL_2' --whitened_layers '2' --epochs 200 --resume model_checkpoints/res18_CUB.pth --use_bn_qcw --lr 5e-4 --lr_decay_epoch 50 --only_load_weights
python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_WL_3' --whitened_layers '3' --epochs 200 --resume model_checkpoints/res18_CUB.pth --use_bn_qcw --lr 5e-4 --lr_decay_epoch 50 --only_load_weights
python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_WL_4' --whitened_layers '4' --epochs 200 --resume model_checkpoints/res18_CUB.pth --use_bn_qcw --lr 5e-4 --lr_decay_epoch 50 --only_load_weights
python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_WL_5' --whitened_layers '5' --epochs 200 --resume model_checkpoints/res18_CUB.pth --use_bn_qcw --lr 5e-4 --lr_decay_epoch 50 --only_load_weights
python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_WL_6' --whitened_layers '6' --epochs 200 --resume model_checkpoints/res18_CUB.pth --use_bn_qcw --lr 5e-4 --lr_decay_epoch 50 --only_load_weights
python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_WL_7' --whitened_layers '7' --epochs 200 --resume model_checkpoints/res18_CUB.pth --use_bn_qcw --lr 5e-4 --lr_decay_epoch 50 --only_load_weights
python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_WL_8' --whitened_layers '8' --epochs 200 --resume model_checkpoints/res18_CUB.pth --use_bn_qcw --lr 5e-4 --lr_decay_epoch 50 --only_load_weights

# -------- concept_data_free : layer-7, different image modes --------
python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_free/ --concepts 'eye,nape' --prefix 'QCW18_SML_WL_7_BLR'  --whitened_layers '7' --concept_image_mode blur   --epochs 200 --resume model_checkpoints/res18_CUB.pth --use_bn_qcw --lr 5e-4 --lr_decay_epoch 50 --only_load_weights
python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_free/ --concepts 'eye,nape' --prefix 'QCW18_SML_WL_7_RDCT' --whitened_layers '7' --concept_image_mode redact --epochs 200 --resume model_checkpoints/res18_CUB.pth --use_bn_qcw --lr 5e-4 --lr_decay_epoch 50 --only_load_weights


############################
# === Analysis Runs ===
############################

# -------- 1) Larger-layers sweep (analysis/larger_layers) --------
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_WL_1_best.pth --concept_dir data/CUB/concept_data_larger/ --whitened_layers '1' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --use_bn_qcw --output_dir 'analysis/larger_layers/WL1_LGR_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_WL_2_best.pth --concept_dir data/CUB/concept_data_larger/ --whitened_layers '2' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --use_bn_qcw --output_dir 'analysis/larger_layers/WL2_LGR_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_WL_3_best.pth --concept_dir data/CUB/concept_data_larger/ --whitened_layers '3' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --use_bn_qcw --output_dir 'analysis/larger_layers/WL3_LGR_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_WL_4_best.pth --concept_dir data/CUB/concept_data_larger/ --whitened_layers '4' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --use_bn_qcw --output_dir 'analysis/larger_layers/WL4_LGR_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_WL_5_best.pth --concept_dir data/CUB/concept_data_larger/ --whitened_layers '5' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --use_bn_qcw --output_dir 'analysis/larger_layers/WL5_LGR_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_WL_6_best.pth --concept_dir data/CUB/concept_data_larger/ --whitened_layers '6' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --use_bn_qcw --output_dir 'analysis/larger_layers/WL6_LGR_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_WL_7_best.pth --concept_dir data/CUB/concept_data_larger/ --whitened_layers '7' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --use_bn_qcw --output_dir 'analysis/larger_layers/WL7_LGR_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_WL_8_best.pth --concept_dir data/CUB/concept_data_larger/ --whitened_layers '8' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --use_bn_qcw --output_dir 'analysis/larger_layers/WL8_LGR_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics

# -------- 2) Image-mode tests (analysis/image_mode_tests) --------

# 2-a  Baseline crop-trained (no image_state flag => crop)
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_SML_WL_7_best.pth  --concept_dir data/CUB/concept_data_free/  --whitened_layers '7'  --hl_concepts 'eye,nape'  --run_purity  --topk_images  --k 10  --use_bn_qcw  --output_dir 'analysis/image_mode_tests/WL7_SML_CROP_METRICS'  --auc_max  --energy_ratio  --masked_auc  --rank_metrics

# 2-b  Blur-trained, eval **with blur**
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_SML_WL_7_BLR_best.pth  --concept_dir data/CUB/concept_data_free/  --whitened_layers '7'  --hl_concepts 'eye,nape'  --run_purity  --topk_images  --k 10  --use_bn_qcw  --output_dir 'analysis/image_mode_tests/WL7_SML_BLR_METRICS'  --auc_max  --energy_ratio  --masked_auc  --rank_metrics  --image_state blur

# 2-c  Blur-trained, eval **on full images** (no flag)
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_SML_WL_7_BLR_best.pth  --concept_dir data/CUB/concept_data_free/  --whitened_layers '7'  --hl_concepts 'eye,nape'  --run_purity  --topk_images  --k 10  --use_bn_qcw  --output_dir 'analysis/image_mode_tests/WL7_SML_BLR2NORMAL_METRICS'  --auc_max  --energy_ratio  --masked_auc  --rank_metrics

# 2-d  Redact-trained, eval **with redact**
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_SML_WL_7_RDCT_best.pth --concept_dir data/CUB/concept_data_free/  --whitened_layers '7'  --hl_concepts 'eye,nape'  --run_purity  --topk_images  --k 10  --use_bn_qcw  --output_dir 'analysis/image_mode_tests/WL7_SML_RDCT_METRICS' --auc_max  --energy_ratio  --masked_auc  --rank_metrics  --image_state redact

# 2-e  Redact-trained, eval **on full images** (no flag)
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_SML_WL_7_RDCT_best.pth --concept_dir data/CUB/concept_data_free/  --whitened_layers '7'  --hl_concepts 'eye,nape'  --run_purity  --topk_images  --k 10  --use_bn_qcw  --output_dir 'analysis/image_mode_tests/WL7_SML_RDCT2NORMAL_METRICS' --auc_max  --energy_ratio  --masked_auc  --rank_metrics


python train_basic.py --main_data data/CUB/main_data/ --concept_data data/CUB/concept_data_larger_flat/ --concepts 'has_back_color::black, has_back_color::blue, has_back_color::white, has_back_color::yellow, has_belly_color::black, has_belly_color::blue, has_belly_color::red, has_belly_color::white, has_belly_color::yellow, has_bill_length::longer_than_head, has_bill_length::shorter_than_head, has_eye_color::black, has_eye_color::red, has_eye_color::white, has_leg_color::black, has_leg_color::red, has_leg_color::yellow, has_nape_color::blue, has_nape_color::green, has_nape_color::red, has_nape_color::white, has_size::medium_(9_-16_in), has_size::very_small(3_-_5_in), has_tail_shape::fan-shaped_tail, has_tail_shape::forked_tail, has_tail_shape::pointed_tail, has_tail_shape::squared_tail, has_throat_color::black, has_throat_color::red, has_throat_color::white, has_throat_color::yellow, has_upper_tail_color::black, has_upper_tail_color::blue, has_upper_tail_color::red, has_upper_tail_color::white, has_upper_tail_color::yellow' --prefix 'oldCW18_LGR_WL_7' --whitened_layers '7' --epochs 100 --resume model_checkpoints/res18_CUB.pth --only_load_weights



python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger_flat/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_OG_WL_1' --whitened_layers '1' --epochs 100 --resume model_checkpoints/res18_CUB.pth --only_load_weights --model resnet --depth 18
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_OG_WL_1_best.pth --concept_dir data/CUB/concept_data_larger_free/ --whitened_layers '1' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --output_dir 'analysis/large18_free_layers/WL6_FREE_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics --model resnet --depth 18

python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger_flat/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_OG_WL_2' --whitened_layers '2' --epochs 100 --resume model_checkpoints/res18_CUB.pth --only_load_weights --model resnet --depth 18
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_OG_WL_2_best.pth --concept_dir data/CUB/concept_data_larger_free/ --whitened_layers '2' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --output_dir 'analysis/large18_free_layers/WL6_FREE_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics --model resnet --depth 18

python train_qcw.py --data_dir data/CUB/main_data/ --concept_dir data/CUB/concept_data_larger_flat/ --concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --prefix 'QCW18_LGR_OG_WL_3' --whitened_layers '3' --epochs 100 --resume model_checkpoints/res18_CUB.pth --only_load_weights --model resnet --depth 18
python plot_functions.py --model_checkpoint model_checkpoints/QCW18_LGR_OG_WL_3_best.pth --concept_dir data/CUB/concept_data_larger_free/ --whitened_layers '3' --hl_concepts 'back,beak,belly,eye,general,leg,nape,tail,throat' --run_purity --topk_images --k 10 --output_dir 'analysis/large18_free_layers/WL6_FREE_METRICS' --auc_max --energy_ratio --masked_auc --rank_metrics --model resnet --depth 18