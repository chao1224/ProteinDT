cd ../../examples/downstream_Editing



pretrained_mode_list=(
    # optimal ones
    ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10
    ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5
    ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-InfoNCE-0.1-batch-9-gpu-8-epoch-5
)
protein_max_sequence_len=50
text_max_sequence_len=512



##### Hyperparameters for downstream-Editing
editing_task_list=(Villin Pin1)
text_prompt_id_list=(101 102 201 202)
protein_max_sequence_len=50
export time=5
export partition=xxx


editing_task_list=(alpha beta)
text_prompt_id_list=(101 201)
protein_max_sequence_len=512
export time=5
export partition=xxx




step_05_folder_template_list=(step_05_AE_1e-3_3 step_05_AE_5e-3_3)


lambda_value_list=(0.1 0.5 0.9)
num_repeat_list=(16)
oracle_mode_list=(protein text)
temperature_list=(0.1 0.5 1 2)
batch_size=8
num_workers=0


for pretrained_mode in ${pretrained_mode_list[@]}; do
for step_05_folder_template in "${step_05_folder_template_list[@]}"; do

    for editing_task in "${editing_task_list[@]}"; do
    for text_prompt_id in "${text_prompt_id_list[@]}"; do
    for lambda_value in "${lambda_value_list[@]}"; do
    for num_repeat in "${num_repeat_list[@]}"; do
    for oracle_mode in "${oracle_mode_list[@]}"; do
    for temperature in "${temperature_list[@]}"; do

        pretrained_folder="../../output/${pretrained_mode}"
        step_05_folder="../../output/${pretrained_mode}/${step_05_folder_template}"

        local_model_dir="${step_05_folder}/downstream_Editing_latent_optimization/${editing_task}_prompt_${text_prompt_id}_lambda_${lambda_value}_num_repeat_${num_repeat}_oracle_${oracle_mode}_T_${temperature}"
        local_output_file="${local_model_dir}/step_01_editing.out"
        local_output_text_file="${local_model_dir}/step_01_editing.txt"


        if test -f "$local_output_text_file"; then
            # ls $local_output_file
            # tail "$local_output_file"
            echo "exists."
            continue
        fi
        echo "not exists."
        tail "$local_output_file"

        mkdir -p "$local_model_dir"

        sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=optimization_"$time" \
        --output="$local_output_file" \
        ./run_step_01_editing_latent_optimization.sh \
        --num_workers=${num_workers} \
        --batch_size=${batch_size} \
        --pretrained_folder=${pretrained_folder} \
        --step_05_folder=${step_05_folder}  \
        --editing_task=${editing_task} \
        --text_prompt_id=${text_prompt_id} \
        --lambda_value=${lambda_value} \
        --num_repeat=${num_repeat} \
        --oracle_mode=${oracle_mode} \
        --temperature=${temperature} \
        --output_folder=${local_model_dir} \
        --protein_max_sequence_len=${protein_max_sequence_len} \
        --output_text_file_path=${local_output_text_file}





        local_output_file="${local_model_dir}/step_01_evaluate.out"

        if test -f "${local_model_dir}/step_01_editing.txt"; then
        if ! test -f "${local_model_dir}/step_01_evaluate.txt"; then
            echo "not exist"
            echo "$local_output_file"

            sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=opt_eval_"$time" \
            --output="$local_output_file" \
            ./run_step_01_evaluate_stability.sh \
            --num_workers=${num_workers} \
            --batch_size=${batch_size} \
            --editing_task=${editing_task} \
            --text_prompt_id=${text_prompt_id} \
            --output_folder=${local_model_dir} \
            --output_text_file_path=${local_output_text_file}
        fi
        fi





        local_output_file="${local_model_dir}/step_01_evaluate.out"

        if test -f "${local_model_dir}/step_01_editing.txt"; then
        if ! test -f "${local_model_dir}/step_01_evaluate.txt"; then
            echo "not exist"
            echo "$local_output_file"

            sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=opt_eval_"$time" \
            --output="$local_output_file" \
            ./run_step_01_evaluate_structure.sh \
            --num_workers=${num_workers} \
            --batch_size=${batch_size} \
            --editing_task=${editing_task} \
            --text_prompt_id=${text_prompt_id} \
            --output_folder=${local_model_dir} \
            --output_text_file_path=${local_output_text_file}
        fi
        fi

    done
    done
    done
    done
    done
    done

done
done
