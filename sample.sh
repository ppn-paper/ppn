model_brats="--model_path evaluations/BraTS/model_2.02m.pt --testset_path evaluations/BraTS/BraTS.npz --attention_resolutions 30 --class_cond False --learn_sigma True --noise_schedule cosine --image_size 240 --num_channels 32 --num_res_blocks 3 --channel_mult 1,2,2,4,4 --use_ddim True"

sampleNum=32

common_params="python -m scripts.image_sample --work_dir working/sampling $model_brats --num_samples $sampleNum --batch_size 16 --show_progress True --timestep_respacing ddim100 --num_timesteps -1"

exec_cmd() {
    echo "--------------------------------------------"
    echo $1 | tr -s ' '  # Print command
    echo "--------------------------------------------"
    eval $1       # run the command
}

exec_cmd "$common_params  --acceleration 4  --sampleType PPN"
exec_cmd "$common_params  --acceleration 8  --sampleType PPN"
exec_cmd "$common_params  --acceleration 16  --sampleType PPN"

exec_cmd "$common_params  --acceleration 4  --sampleType SONG"
exec_cmd "$common_params  --acceleration 8  --sampleType SONG"
exec_cmd "$common_params  --acceleration 16  --sampleType SONG"

exec_cmd "$common_params  --acceleration 4  --sampleType DDNM"
exec_cmd "$common_params  --acceleration 8  --sampleType DDNM"
exec_cmd "$common_params  --acceleration 16  --sampleType DDNM"

exec_cmd "$common_params  --acceleration 4  --sampleType DPS"
exec_cmd "$common_params  --acceleration 8  --sampleType DPS"
exec_cmd "$common_params  --acceleration 16  --sampleType DPS"
