# 🤖 When Benchmarks Talk

This is the official code release for the Findings of ACL 2025 paper "When Benchmarks Talk: Re-Evaluating Code LLMs with Interactive Feedback" (https://arxiv.org/abs/2502.18413).

You can find our collected data here: https://huggingface.co/datasets/janepan9917/when-benchmarks-talk

Details about how to use this codebase are below.

If you use this codebase or data, we'd appreciate it if you cite our work!

```bibtex
@misc{pan2025benchmarkstalkreevaluatingcode,
      title={When Benchmarks Talk: Re-Evaluating Code LLMs with Interactive Feedback}, 
      author={Jane Pan and Ryan Shar and Jacob Pfau and Ameet Talwalkar and He He and Valerie Chen},
      year={2025},
      eprint={2502.18413},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2502.18413}, 
}
```

## 🎯 The Big Picture

Our experiment consists of four phases:

1. **🔧 Configuration Setup** - Define your models and runtime settings
2. **💬 Multi-turn Generation** - Create interactive conversations between AI models  
3. **🧪 Evaluation Magic** - Test and analyze the generated code
4. **📊 Results & Insights** - Aggregate data and create beautiful visualizations

---

## ⚙️ Configuration Made Easy

### 📁 Configuration Files

All configuration happens through YAML configuration files stored in the `config/` directory.
An example of a configuration for running APPS experiments using gpt-4o as code model and sonnet 3.5 as the user model providing paragraph style feedback:
```
configs/interaction_4o_w_sonnet-3.5_user/ir_nl_feedback_paragraph.yaml
```

### 🎛️ Configuration Breakdown

#### 🔄 Interaction Settings (`interaction`)
- **`feedback_setting`**: Type of feedback mechanism (currently supports `iter_refinement`)
- **`iter_refinement`**: Number of feedback rounds between models
- **`feedback_type`**: Format of feedback (`free_response_answer`, `nl_feedback`, or `test_case`)
- **`n_samples`**: How many times to run the generation

#### 📚 Dataset Configuration (`dataset`)
- **`input_transformation_type`**: How we summarize the input data
- **`solution_info_type`**: What solution info to provide (we use `"full_solution"`)
- **`shuffle`**: Boolean value, if the question order should be randomized first 

#### 👥 User Model Settings (`user_model`)
The model that provides feedback to the code generator:
- **`model_name`**: Which AI model to use (we mostly use `"claude-3.5-sonnet"`)
- **`api_key`**: Environment variable with the model's API key 
- **`prompt_fn`**: Prompt template file (check `prompts/user/` for examples)
- **`include_history`**: Whether to include the full conversation history

#### 🤖 Assistant Model Settings (`asst_model`)
The code-generating AI assistant:
- **`model_name`**: The coding assistant model
- **`api_key`**: Environment with the model's API key 
- **`prompt_fn`**: Prompt template (see `prompts/asst/` for inspiration)
- **`include_history`**: Whether to include the full conversation history

---

## 🌟 Generation Process

### 🛠️ Environment Setup

Get your environment ready in two simple commands:

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate and you're ready to go! 
conda activate code-uncert
```

### 📊 Input Data Structure

Our datasets live in the `data/` directory with this neat organization:

```
dataset_name/test/difficulty/summarization_type/question_id.json
```

#### 🔑 Essential JSON Fields
Every question file contains these crucial elements:
- **Common fields**: `input`, `orig_input`, `qid`, `solutions`, `split`, `task`, `test_cases`, `transformation_type`
- **APPS/LiveCodeBench**: Additional `difficulty` field
- **ClassEval**: Extra goodies like `import_statement`, `method_info`, `test_classes`

### 🎬 Running the Generation

Time for the main event! To run the main script, we must provide the configuration file, the dataset directory, and a directory to store the outputs.

Here's an example of how to generate APPS interview data:
```bash
python -m src.main \
    --config_fn configs/interaction_4o_w_sonnet-3.5_user/ir_nl_feedback_paragraph.yaml \
    --data_dir data/apps/interview/test \
    --output_dir /data/some/location/.../interaction_4o_w_sonnet_3.5_user/interview/paragraph
```

**Magic happens here!** ✨ Your results will appear in:
```
/data/some/location/.../apps/interaction_4o_w_sonnet_3.5_user/interview/paragraph/raw_output
```

Each subdirectory corresponds to a unique `question_id` 

---

## 🧪 Evaluation Extravaganza

Time to see how well our models performed! Each dataset has its own evaluation approach, and they all generate a `test_case_results.json` file for aggregation.

### 📱 APPS Evaluation

For APPS questions, we use the `eval.apps.test_one_solution` module. This class requires the qid as an argument. Since the qids are ints, we can add a simple bash loop to do all questions.

```bash
# Loop through questions like a pro! 
for qid in {0..100}
do
    output_dir="/data/some/location/.../apps/interaction_4o_w_sonnet_3.5_user/interview/paragraph"
    python -m eval.apps.test_one_solution -d \
        --output_dir $output_dir \
        --qid $qid \
        --n_samples 4
done
```

### 🔴 LiveCodeBench Evaluation

LiveCodeBench is run with `eval.livecodebench.test_solutions`:

```bash
output_dir="/data/some/location/.../livecodebench/interaction_4o_w_sonnet_3.5_user/easy/ir_nl_feedback_sentence"

python -m eval.livecodebench.test_solutions \
    --output_dir $output_dir \
    --n_samples 4
```

### 🏫 ClassEval Evaluation

ClassEval is special - it needs a two steps:

**Step 1: Test the Classes**
```bash
cd eval/class_eval

config="interaction_4o_w_sonnet-3.5_user"
setting="ir_code_feedback"
output_dir="/data/some/location/.../classeval/${config}/${setting}"

python test_classes.py \
    --output_dir $output_dir \
    --n_steps 5
```

**Step 2: Generate Test Case JSONs**
```bash
cd eval/class_eval
config="interaction_4o_w_sonnet-3.5_user"
setting="ir_code_feedback"
output_dir="/data/some/location/.../classeval/${config}/${setting}"
python make_qid_test_case_jsons.py \
    --dir $output_dir \
    --n_steps 5  # Match the experiment config
```

---

## Aggregation

After running all evaluations, you'll have beautiful `test_case_results.json` files ready for aggregation! 
To aggregate the results over the questions, run the `aggregate_test_case_results.py` script:
```bash
output_dir="/data/some/location/.../${dataset}/${config}/${setting}"
stats_dir="/data/some/other/place/..../stats/${dataset}/${config}/${setting}"

cd analysis
python aggregate_test_case_results.py \
    --dir $output_dir \
    --stats_dir $stats_dir
```
This will create a `stats` directory at `$output_dir` which contains the aggregated results for each step -- e.g. `$output_dir/stats/0` for step 0 results.

This will also create an `all_stats.tsv` file at `$stats_dir` to contain the aggregated final results. 

---

*💡 Pro Tip: Each evaluation method generates consistent output formats, making aggregation and analysis a breeze!*
