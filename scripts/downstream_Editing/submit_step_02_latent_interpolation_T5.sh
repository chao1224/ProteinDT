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
editing_task_list=(peptide_binding)
text_prompt_id_list=(101 201)
export time=5
export partition=xxx
# specifically for inference_02_aggregated
batch_size=1



theta_list=(0.1 0.5 0.9)
num_repeat_list=(16)
oracle_mode_list=(protein text condition)
AR_generation_mode_list=(01 02)
AR_condition_mode_list=(aggregated expanded)




##### Hyperparameters for step-04
prob_unconditional_list=(0)
num_workers=0
lr_list=(1e-4 1e-5)
epochs_list=(10)
decoder_distribution=T5Decoder
score_network_type=T5Base
hidden_dim_list=(16 32)


for pretrained_mode in "${pretrained_mode_list[@]}"; do
for lr in "${lr_list[@]}"; do
for hidden_dim in "${hidden_dim_list[@]}"; do
for prob_unconditional in "${prob_unconditional_list[@]}"; do
for epochs in "${epochs_list[@]}"; do

    for editing_task in "${editing_task_list[@]}"; do
    for text_prompt_id in "${text_prompt_id_list[@]}"; do
    for theta in "${theta_list[@]}"; do
    for num_repeat in "${num_repeat_list[@]}"; do
    for oracle_mode in "${oracle_mode_list[@]}"; do

    for AR_generation_mode in "${AR_generation_mode_list[@]}"; do
    for AR_condition_mode in "${AR_condition_mode_list[@]}"; do

        pretrained_folder="../../output/${pretrained_mode}"
        step_04_folder="../../output/${pretrained_mode}/step_04_${decoder_distribution}_${score_network_type}_lr_${lr}_hidden_${hidden_dim}_e_${epochs}_unconditional_${prob_unconditional}"
        
        local_model_dir="${step_04_folder}/downstream_Editing_latent_interpolation_${editing_task}/prompt_${text_prompt_id}_theta_${theta}_num_repeat_${num_repeat}_oracle_${oracle_mode}_inference_${AR_generation_mode}_${AR_condition_mode}"
        local_output_file="${local_model_dir}/step_02_editing.out"
        local_output_text_file="${local_model_dir}/step_02_editing.txt"

        if test -f "$local_output_text_file"; then
            echo "exists."
            continue
        fi
        echo "not exists."

        mkdir -p "$local_model_dir"

        sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=Edit_T5_"$time" \
        --output="$local_output_file" \
        ./run_step_02_editing_latent_interpolation.sh \
        --num_workers=${num_workers} \
        --decoder_distribution=${decoder_distribution} \
        --score_network_type=${score_network_type} \
        --hidden_dim=${hidden_dim} \
        --batch_size=${batch_size} \
        --pretrained_folder=${pretrained_folder} \
        --step_04_folder=${step_04_folder}  \
        --editing_task=${editing_task} \
        --text_prompt_id=${text_prompt_id} \
        --theta=${theta} \
        --num_repeat=${num_repeat} \
        --oracle_mode=${oracle_mode} \
        --AR_generation_mode=${AR_generation_mode} \
        --AR_condition_mode=${AR_condition_mode} \
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
done
done
done
done
