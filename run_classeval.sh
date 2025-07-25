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
    --data_dir data/class_eval/test \
    --output_dir ${output_dir}

# Classeval uses different testing code so re need to run it in two steps:
# 1. run the tests with the original classeval code
# 2. transform the results to the format expected by our analysis code
cd eval/class_eval
n_steps=5  # Match the experiment config

python test_classes.py \
    --output_dir $output_dir \
    --n_steps $n_steps
python make_qid_test_case_jsons.py \
    --dir $output_dir \
    --n_steps $n_steps  

# aggregate test case results and generate classifications with gpt-4o
cd ../../analysis
python aggregate_test_case_results.py \
    --dir $output_dir \
    --stats_dir $stats_dir

python save_llm_eval_classifications.py \
    --dir $output_dir \