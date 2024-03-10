cd ../../examples/downstream_Text2Protein



pretrained_mode_list=(
    # optimal ones
    ProteinDT/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10
    ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-EBM_NCE-0.1-batch-9-gpu-8-epoch-5
    ProteinDT/ProtBERT_BFD-512-1e-5-1e-1-text-512-1e-5-1e-1-InfoNCE-0.1-batch-9-gpu-8-epoch-5
)
protein_max_sequence_len=512
text_max_sequence_len=512

# Hyperparameters for downstream-Text2Protein
num_repeat_list=(16)
use_facilitator_list=(use_facilitator no_use_facilitator)
batch_size=8
AR_generation_mode_list=(01 02)


# Hyperparameters for step-04
prob_unconditional_list=(0 0.1)
num_workers=0
lr_list=(1e-4 1e-5)
epochs_list=(10 30 50)
decoder_distribution=T5Decoder
score_network_type=T5Base
alpha_1=1
alpha_2=0
hidden_dim_list=(16 32)


export time=5
export partition=xxx


for pretrained_mode in "${pretrained_mode_list[@]}"; do
for lr in "${lr_list[@]}"; do
for hidden_dim in "${hidden_dim_list[@]}"; do
for prob_unconditional in "${prob_unconditional_list[@]}"; do
for epochs in "${epochs_list[@]}"; do
for num_repeat in "${num_repeat_list[@]}"; do
for use_facilitator in "${use_facilitator_list[@]}"; do
for AR_generation_mode in "${AR_generation_mode_list[@]}"; do


    pretrained_folder="../../output/${pretrained_mode}"
    step_04_folder="../../output/${pretrained_mode}/step_04_${decoder_distribution}_${score_network_type}_lr_${lr}_hidden_${hidden_dim}_e_${epochs}_unconditional_${prob_unconditional}"

    local_model_dir="${step_04_folder}/downstream_Text2Protein/num_repeat_${num_repeat}_${use_facilitator}_inference_${AR_generation_mode}"
    local_output_file="${local_model_dir}/step_02_inference.out"
    local_output_text_file="${local_model_dir}/step_02_inference.txt"


    if test -f "$local_output_text_file"; then
        echo "$local_output_text_file exists."
        tail "$local_output_file"
        continue
    fi
    echo "$local_output_text_file not exists."





    mkdir -p "$local_model_dir"

    sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=T5_"$time" \
    --output="$local_output_file" \
    ./run_step_02_inference_ProteinDT.sh \
    --num_workers=${num_workers} \
    --decoder_distribution=${decoder_distribution} \
    --score_network_type=${score_network_type} \
    --hidden_dim=${hidden_dim} \
    --batch_size=${batch_size} \
    --pretrained_folder=${pretrained_folder} \
    --step_04_folder=${step_04_folder}  \
    --num_repeat=${num_repeat} \
    --${use_facilitator} \
    --AR_generation_mode=${AR_generation_mode} \
    --output_text_file_path="$local_output_text_file"


done
done
done
done
done
done
done
done
