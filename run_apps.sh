# build and activate env
conda env create -f environment.yml
conda activate code-uncert


output_dir="SPECIFY_YOUR_OUTPUT_DIR_HERE"
stats_dir="SPECIFY_YOUR_STATS_DIR_HERE"

# To get the generations
# Make sure you update the env variables to have the API keys:
#     - CLAUDE_KEY for claude key
#     - OPENAI_KEY for openai key
python -m src.main \
    --config_fn configs/interaction_4o_w_sonnet-3.5_user/ir_nl_feedback_paragraph.yaml \
    --data_dir data/apps/interview/test \
    --output_dir ${output_dir}

# For apps our evaluation code runs on one question at a time
# so we loop through the question ids
for qid in {0..100}
do
    python -m eval.apps.test_one_solution -d \
        --output_dir $output_dir \
        --qid $qid \
        --n_samples 4
done


# aggregate test case results and generate classifications with gpt-4o
cd analysis
python aggregate_test_case_results.py \
    --dir $output_dir \
    --stats_dir $stats_dir

python save_llm_eval_classifications.py \
    --dir $output_dir \