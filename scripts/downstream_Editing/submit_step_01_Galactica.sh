cd ../../examples/downstream_Editing



##### Hyperparameters for downstream-Editing
batch_size=4
num_workers=0


editing_task_list=(Villin Pin1)
text_prompt_id_list=(101 102 201 202)
export time=5
export partition=xxx


editing_task_list=(alpha beta)
text_prompt_id_list=(101 201)
export time=5
export partition=xxx



for editing_task in "${editing_task_list[@]}"; do
for text_prompt_id in "${text_prompt_id_list[@]}"; do

    local_model_dir="../../output/downstream_Editing/Galactica/${editing_task}_text_prompt_id_${text_prompt_id}"
    local_output_file="${local_model_dir}/step_01_editing.out"
    local_output_text_file="${local_model_dir}/step_01_editing.txt"
    
    if test -f "$local_output_file"; then
        echo "$local_output_file exists."
        continue
    fi
    echo "$local_output_file not exists."


    mkdir -p "$local_model_dir"

    sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=Galactica_"$time" \
    --output="$local_output_file" \
    ./run_step_01_editing_Galactica.sh \
    --num_workers=${num_workers} \
    --batch_size=${batch_size} \
    --editing_task=${editing_task} \
    --text_prompt_id=${text_prompt_id} \
    --output_folder=${local_model_dir} \
    --output_text_file_path=${local_output_text_file}


    local_output_file="${local_model_dir}/step_01_evaluate.out"

    sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=Galactica_eval_"$time" \
    --output="$local_output_file" \
    ./run_step_01_evaluate_stability.sh \
    --num_workers=${num_workers} \
    --batch_size=${batch_size} \
    --editing_task=${editing_task} \
    --text_prompt_id=${text_prompt_id} \
    --output_folder=${local_model_dir} \
    --output_text_file_path=${local_output_text_file}





    local_output_file="${local_model_dir}/step_01_evaluate.out"

    if ! test -f "${local_model_dir}/step_01_evaluate.txt"; then
        echo "not exist"
        echo "$local_output_file"

        sbatch --gres=gpu:1 -n 8 --mem 32G --nodes 1 -t "$time":59:00 --partition="$partition" --job-name=Galactica_eval_"$time" \
        --output="$local_output_file" \
        ./run_step_01_evaluate_structure.sh \
        --num_workers=${num_workers} \
        --batch_size=${batch_size} \
        --editing_task=${editing_task} \
        --text_prompt_id=${text_prompt_id} \
        --output_folder=${local_model_dir} \
        --output_text_file_path=${local_output_text_file}
    fi


done
done
