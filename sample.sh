model_brats="--model_path evaluations/BraTS/model_2.02m.pt --testset_path evaluations/BraTS/BraTS.npz --attention_resolutions 30 --class_cond False --learn_sigma True --noise_schedule cosine --image_size 240 --num_channels 32 --num_res_blocks 3 --channel_mult 1,2,2,4,4 --use_ddim True"

sampleNum=4

cmd_brats="python -m scripts.image_sample --work_dir working/sampling $model_brats --num_samples $sampleNum --batch_size 32 --timestep_respacing ddim1000 --acceleration 4 --show_progress True --sampleType PPN"

exec_cmd() {
    echo "--------------------------------------------"
    echo $1 | tr -s ' '  # Print command
    echo "--------------------------------------------"
    eval $1       # run the command
}

exec_cmd "$cmd_brats"

