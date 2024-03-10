cd ../../examples/downstream_Editing



pretrained_mode_list=(
    # optimal ones
    ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10
    ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5
    ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-InfoNCE-0.1-batch-9-gpu-8-epoch-5
)
protein_max_sequence_len=512
text_max_sequence_len=512

##### Hyperparameters for downstream-Editing
editing_task_list=(Villin Pin1)
text_prompt_id_list=(101 102 201 202)
export time=5
export partition=xxx


editing_task_list=(alpha beta)
text_prompt_id_list=(101 201)
export time=5
export partition=xxx




theta_list=(0.1 0.5 0.9)
num_repeat_list=(16)
oracle_mode_list=(protein text condition)
batch_size=16



##### Hyperparameters for step-04
prob_unconditional_list=(0)
num_workers=0
lr_list=(1e-4 1e-5)
decoder_distribution=MultinomialDiffusion
hidden_dim_list=(16 32)
# SDE_sampling_mode_list=(simplified weighted)
SDE_sampling_mode_list=(simplified)





score_network_type_list=(BertBase)
epochs_list=(10)

score_network_type_list=(RNN)
epochs_list=(50)




for pretrained_mode in "${pretrained_mode_list[@]}"; do
for score_network_type in "${score_network_type_list[@]}"; do
for lr in "${lr_list[@]}"; do
for hidden_dim in "${hidden_dim_list[@]}"; do
for prob_unconditional in "${prob_unconditional_list[@]}"; do
for epochs in "${epochs_list[@]}"; do

    for editing_task in "${editing_task_list[@]}"; do
    for text_prompt_id in "${text_prompt_id_list[@]}"; do
    for theta in "${theta_list[@]}"; do
    for num_repeat in "${num_repeat_list[@]}"; do
    for SDE_sampling_mode in "${SDE_sampling_mode_list[@]}"; do
    for oracle_mode in "${oracle_mode_list[@]}"; do

        pretrained_folder="../../output/${pretrained_mode}"
        step_04_folder="../../output/${pretrained_mode}/step_04_${decoder_distribution}_${score_network_type}_lr_${lr}_hidden_${hidden_dim}_e_${epochs}_unconditional_${prob_unconditional}"

        local_model_dir="${step_04_folder}/downstream_Editing_latent_interpolation_${editing_task}/prompt_${text_prompt_id}_theta_${theta}_num_repeat_${num_repeat}_${SDE_sampling_mode}_oracle_${oracle_mode}"
        local_output_file="${local_model_dir}/step_01_editing.out"
        local_output_text_file="${local_model_dir}/step_01_editing.txt"


        if test -f "$local_output_text_file"; then
            echo "exists."
            continue
        fi
        echo "not exists."

        mkdir -p "$local_model_dir"

        sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=Edit_Multi_"$time" \
        --output="$local_output_file" \
        ./run_step_01_editing_latent_interpolation.sh \
        --num_workers=${num_workers} \
        --decoder_distribution=${decoder_distribution} \
        --score_network_type=${score_network_type} \
        --hidden_dim=${hidden_dim} \
        --batch_size=${batch_size} \
        --pretrained_folder=${pretrained_folder} \
        --step_04_folder=${step_04_folder}  \
        --editing_task=${editing_task} \
        --text_prompt_id=${text_prompt_id} \
        --theta=$theta \
        --num_repeat=${num_repeat} \
        --SDE_sampling_mode=${SDE_sampling_mode} \
        --oracle_mode=${oracle_mode} \
        --output_folder=${local_model_dir} \
        --output_text_file_path=${local_output_text_file}




        local_output_file="${local_model_dir}/step_01_evaluate.out"

        if test -f "${local_model_dir}/step_01_editing.txt"; then
        if ! test -f "${local_model_dir}/step_01_evaluate.txt"; then
            echo "not exist"
            echo "$local_output_file"
                
            sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=Multi_eval_"$time" \
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
                
            sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=Multi_eval_"$time" \
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
done
done
done
done
