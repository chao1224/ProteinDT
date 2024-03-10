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
editing_task_list=(peptide_binding)
text_prompt_id_list=(101 201)
export time=5
export partition=xxx




step_05_folder_template_list=(step_05_AE_1e-3_3 step_05_AE_5e-3_3)


lambda_value_list=(0.1 0.5 0.9)
num_repeat_list=(16)
oracle_mode_list=(protein text)
temperature_list=(0.1 0.5 1 2)
batch_size=4
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
        local_output_file="${local_model_dir}/step_02_editing.out"
        local_output_text_file="${local_model_dir}/step_02_editing.txt"

        if test -f "$local_output_text_file"; then
            echo "exists."
            continue
        fi
        echo "not exists."

        mkdir -p "$local_model_dir"

        sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=optimization_"$time" \
        --output="$local_output_file" \
        ./run_step_02_editing_latent_optimization.sh \
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
        --output_text_file_path=${local_output_text_file}


    done
    done
    done
    done
    done
    done

done
done
