# Examples

# These are examples on how to run the training and evaluation of the system.

# Model Training

CUDA_VISIBLE_DEVICES=0 python run_training.py --save_dir ./models --system local --data_file v10 --json ./models/tf-1527876409-imu_v9-birnn-fc1_512-lstm2_512-idrop2-relu-norm_ori_acc_smpl-auxloss_acc/config.json

# Trained Model Refining
CUDA_VISIBLE_DEVICES=1 python run_training.py --save_dir ./models --model_id 1579780111 --system local --data_file v10 --norm_ori --norm_acc --norm_smpl --use_acc_loss --finetune_train_data imu_own_training.npz --finetune_valid_data imu_own_test.npz


# Training Dataset
/home/lisca/data/dip/synthetic_60fps/AMASS_ACCAD/Male2Walking_c3dB17_SB__SB2__SB__SB_Walk_SB_to_SB_hop_SB_to_SB_walk_SB_a_dynamics.pkl

# Refining Dataset
/home/lisca/data/dip/dip_imu_and_others/DIP_IMU/s_01/01.pkl

# Evaluation (online)
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --system local --data_file own --model_id 1585846831 --save_dir ./models --eval_dir ./evaluation_results/ --datasets dip-imu --save_predictions --verbose 1 --past_frames 10 30 50 --future_frames 1 3 5 7 9

# Evaluation live
/home/lisca/code/dip18/train_and_eval/evaluation_results/tf-1583963318--amass2imusim-birnn-fc1_512-lstm2_512-idrop2-relunorm_ori_acc_smpl-auxloss_acc/test_our_data_all_frames.npz
