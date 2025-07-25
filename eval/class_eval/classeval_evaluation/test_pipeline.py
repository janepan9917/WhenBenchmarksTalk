import shutil
import time
from func_timeout import func_set_timeout
import importlib
import unittest
import json
import re
import os
import sys
import gc
from scipy.special import comb
# from path_util import PathUtil

from pathlib import Path
from hashlib import sha256

class AutoTest:

    def __init__(self, eval_data_name: Path):
        self.path = eval_data_name.parent.resolve()
        self.eval_data = self.get_eval_data(eval_data_name)
        self.TEAR_DOWN_TARGET_DIR_NAME = "classeval_evaluation"
        self._imported_modules = set()


    def _import_module(self, module_name):
        """Safely import a module and track it"""
        try:
            # Force reload if module was previously imported
            if module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
            else:
                module = importlib.import_module(module_name)
            self._imported_modules.add(module_name)
            return module
        except ImportError as e:
            print(f"Failed to import {module_name}: {e}")
            raise

    def get_eval_data(self, eval_data_name):
        eval_data = {}
        with open(eval_data_name, encoding='utf-8') as file:
            data = json.load(file)
        for item in data:
            eval_data[item['task_id']] = item
        return eval_data

    def gen_py_file(self, test_code_name, code_list, test_code):
        cnt = 0
        for code_snippet in code_list:
            test_code_py = code_snippet + '\n' + test_code
            with open(test_code_name + '_' + str(cnt) + '.py', 'w', encoding='utf-8') as f:
                f.write(test_code_py)
            cnt += 1

    def get_leading_spaces(self, string):
        return len(string) - len(string.lstrip())

    def extract_imports(self, code_snippet):
        pattern = r'^import\s.*$|from\s.*\simport.*$'
        imports = re.findall(pattern, code_snippet, re.MULTILINE)
        return imports

    def extract_code(self, text, model_name):
        text = text.rstrip()
        output_split_identifier_list = ["### Response:", "@@ Response:", "[/INST]"]
        for identifier in output_split_identifier_list:
            if identifier in text:
                text = text.split(identifier)[1]
                break

        if "incoder" in model_name:
            # remove <|/ file |>
            if "<|/ file |>" in text:
                text = text.split("<|/ file |>")[0]
            return text

        else:
            pattern_list = [r"```python(.*?)```", r"```ruby(.*?)```", r"```scss(.*?)```",
                            r"```python(.*?)", r"```(.*?)```", r"\[PYTHON\](.*?)\[/PYTHON\]"]
            for pattern in pattern_list:
                try:
                    code = re.findall(pattern, text, re.S)[0]
                    return code
                except:
                    continue

            code_list = text.split("\n")
            removed_lines = []
            for code_line in code_list:
                if code_line.strip().startswith('class'):
                    break
                elif not code_line.strip().startswith('import') and not code_line.strip().startswith('from'):
                    removed_lines.append(code_line)
            code_list = [line for line in code_list if line not in removed_lines]
            text = '\n'.join(code_list)

            wrong_indent_flag = False
            for code_line in text.split("\n"):
                if code_line.strip().startswith('class'):
                    class_signature_line_leading_spaces = self.get_leading_spaces(code_line)
                    if class_signature_line_leading_spaces != 0:
                        wrong_indent_flag = True
                    break
            if wrong_indent_flag:
                final_code_line_list = []
                for code_line in text.split("\n"):
                    cur_leading_spaces = self.get_leading_spaces(code_line)
                    # Keep the relative indentation unchanged
                    final_code_line_list.append(' ' * (cur_leading_spaces - class_signature_line_leading_spaces) + code_line.lstrip())
                text = '\n'.join(final_code_line_list)
            return text

    def add_static_statement(self, code):
        filtered_code_list = []
        for line in code.split('\n'):
            if '@staticmethod' in line:
                continue
            filtered_code_list.append(line)
        code = '\n'.join(filtered_code_list)
        final_code_list = []
        for line in code.split('\n'):
            if line.strip().startswith('def ') and 'self' not in line and 'cls' not in line and self.get_leading_spaces(line) == 4:
                final_code_list.append('    @staticmethod')
            final_code_list.append(line)
        return '\n'.join(final_code_list)

    def gen_code_list(self, file_path):
        code_list = {}

        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            code_list[item['task_id']] = []
            for predict in item['predict']:
                predict = self.extract_code(predict, str(file_path))
                predict = predict.split('python')
                predict = predict[-1]
                predict = self.add_static_statement(predict)
                predict = '\n'.join(self.eval_data[item['task_id']]['import_statement']) + '\n' + predict
                code_list[item['task_id']].append(predict)
        return code_list

    @func_set_timeout(5)
    def run_unit_test(self, test_code, test_class, model_name):
        module = self._import_module(test_code)
        log_path = Path(self.path, f"{model_name}_log_data.log")
        with open(log_path, 'a+', encoding='utf-8') as f:
            test_suite = unittest.TestLoader().loadTestsFromTestCase(getattr(module, test_class))
            test_result = unittest.TextTestRunner(stream = f).run(test_suite)

        return test_result

    def test(self, code_num, test_code_name, test_classes, model_name):

        result = {}
        # import pdb; pdb.set_trace()
        for i in range(code_num):
            print(f"\tSample {i}")
            test_code = test_code_name + '_' + str(i)
            result[test_code] = {}
            
            for test_class in test_classes:
                res_item = {}
                try:
                    res = self.run_unit_test(test_code, test_class, model_name)
                    res_item['errors'] = len(res.errors)
                    res_item['failures'] = len(res.failures)
                    res_item['testsRun'] = res.testsRun
                    result[test_code][test_class] = res_item
                except Exception as e:
                    res_item['errors'] = 0
                    res_item['failures'] = 0
                    res_item['testsRun'] = 0
                    result[test_code][test_class] = res_item

        return result

    def save_result(self, model_name, result, type):
        save_path = Path(self.path, f"{model_name}_result.json")
        print(f"\tSaving to {save_path.resolve()}")
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4, sort_keys=True)
            f.flush()

    def test_pipeline(self, model_name, gen_file_path):
        try:
            self.tear_down()
            result_dict = {}
            # get generate code list
            code_list = self.gen_code_list(gen_file_path)

            # get test code and generate py file
            for task_id in code_list:
                test_code = self.eval_data[task_id]['test']
                task_code_list = code_list[task_id]

                self.gen_py_file(task_id, task_code_list, test_code)
            
            # run unit test
            time.sleep(2)
            for task_id in code_list:
                print(f"===== EVALUATING {task_id} =====")
                task_code_list = code_list[task_id]
                print(f"\ttask_code_list len: {len(task_code_list)}")
                print(f"\ttest_classes len: {len(self.eval_data[task_id]['test_classes'])}")
                try:
                    result = self.test(len(task_code_list), task_id,
                                    self.eval_data[task_id]['test_classes'], model_name)
                    result_dict[task_id] = result
                except:
                    print(f" ============== failed for {task_id} ==============")
                    # Optionally: force set everything to 0 (we did not do this, original paper didn't either)
                    # result = {}
                    # for i in range(len(task_code_list)):
                    #     test_code = task_id+ '_' + str(i)
                    #     result[test_code] = {}

                    #     for test_class in self.eval_data[task_id]['test_classes']:
                    #         res_item = {}
                    #         res_item['errors'] = 0
                    #         res_item['failures'] = 0
                    #         res_item['testsRun'] = 0
                    #         result[test_code][test_class] = res_item
                    # result_dict[task_id] = result
                    continue 
            
            self.save_result(model_name, result_dict, "class")
        finally:
            self.tear_down()
            time.sleep(5)

    def get_test_answer(self, test_result):
        if test_result['testsRun'] == 0 or test_result['errors'] == test_result['testsRun']:
            return 'error'
        if test_result['errors'] + test_result['failures'] == 0:
            return 'success'
        if test_result['errors'] + test_result['failures'] < test_result['testsRun']:
            return 'partial_success'
        return 'fail'

    def evaluate(self, model_list: list[str]):

        result_dict = {}
        for model_name in model_list:
            model_result_path = Path(self.path, f"{model_name}_result.json")
            with open(model_result_path, 'r') as f:
                model_result = json.load(f)
            result_dict[model_name] = {}
            for task in model_result:
                result_dict[model_name][task] = {}
                for test_num in model_result[task]:
                    temp_result = {"success": 0,
                                   "partial_success": 0, "fail": 0, "error": 0}
                    for test_class in model_result[task][test_num]:
                        if test_class not in result_dict[model_name][task]:
                            result_dict[model_name][task][test_class] = {}
                            result_dict[model_name][task]["TestClass"] = {}
                            result_dict[model_name][task]["TestClass"]["ClassEachTestResult"] = [
                            ]
                            result_dict[model_name][task][test_class]['success'] = 0
                            result_dict[model_name][task][test_class]['partial_success'] = 0
                            result_dict[model_name][task][test_class]['fail'] = 0
                            result_dict[model_name][task][test_class]['error'] = 0
                            result_dict[model_name][task][test_class]["EachTestResult"] = [
                            ]
                            result_dict[model_name][task]["TestClass"]["class_success"] = 0
                            result_dict[model_name][task]["TestClass"]["class_partial_success"] = 0
                            result_dict[model_name][task]["TestClass"]["class_fail"] = 0
                        test_answer = self.get_test_answer(
                            model_result[task][test_num][test_class])
                        result_dict[model_name][task][test_class][test_answer] += 1
                        result_dict[model_name][task][test_class]["EachTestResult"].append(
                            test_answer)
                        temp_result[test_answer] += 1
                    if temp_result['success'] == len(model_result[task][test_num]):
                        result_dict[model_name][task]["TestClass"]["class_success"] += 1
                        result_dict[model_name][task]["TestClass"]["ClassEachTestResult"].append(
                            "class_success")
                    elif temp_result['fail'] == 0 and temp_result['error'] == 0:
                        result_dict[model_name][task]["TestClass"]["class_partial_success"] += 1
                        result_dict[model_name][task]["TestClass"]["ClassEachTestResult"].append(
                            "class_partial_success")
                    else:
                        result_dict[model_name][task]["TestClass"]["class_fail"] += 1
                        result_dict[model_name][task]["TestClass"]["ClassEachTestResult"].append(
                            "class_fail")

        save_path = Path(self.path, f"{str(model_list[0])}_detailed_result.json")
        with open(save_path, 'w') as f:
            json.dump(result_dict, f, indent=4, sort_keys=True)

    def cal_pass_at_k(self, n, k, k_success):
        total_combinations = comb(k, n)
        if k - k_success >= n:
            without_k_success_combinations = comb(k - k_success, n)
        else:
            without_k_success_combinations = 0

        with_k_success_combinations = total_combinations - without_k_success_combinations

        pass_at_k = with_k_success_combinations / total_combinations

        return pass_at_k

    def cal_metrics_pass_at_k(self, model_list, n, k):

        file_path = Path(self.path, f"{str(model_list[0])}_detailed_result.json")
        with open(file_path, 'r') as f:
            test_result = json.load(f)

        result = {}

        for model_name in model_list:
            sum_num = 0
            success_num = 0
            class_success_num = 0
            class_num = 0
            partial_success_num = 0
            partial_success_class_num = 0
            for task in test_result[model_name]:
                class_num += 1
                for test_class in test_result[model_name][task]:
                    try:
                        if test_result[model_name][task][test_class]['success'] != 0:
                            pass_at_k = self.cal_pass_at_k(
                                n, k, test_result[model_name][task][test_class]['success'])
                            success_num += pass_at_k
                        if test_result[model_name][task][test_class]['success'] + test_result[model_name][task][test_class]['partial_success'] != 0:
                            pass_at_k = self.cal_pass_at_k(
                                n, k, test_result[model_name][task][test_class]['success'] + test_result[model_name][task][test_class]['partial_success'])
                            partial_success_num += pass_at_k
                        sum_num += 1
                    except:
                        if test_result[model_name][task][test_class]['class_success'] != 0:
                            pass_at_k = self.cal_pass_at_k(
                                n, k, test_result[model_name][task][test_class]['class_success'])
                            class_success_num += pass_at_k
                        k_success = test_result[model_name][task][test_class]['class_success'] + \
                            test_result[model_name][task][test_class]['class_partial_success']
                        if k_success != 0:
                            pass_at_k = self.cal_pass_at_k(n, k, k_success)
                            partial_success_class_num += pass_at_k
            if sum_num == 0:
                fun_succ = 0
                fun_part = 0
            else:
                fun_succ = success_num/sum_num
                fun_part = partial_success_num / sum_num
            if class_num == 0:
                class_succ = 0
                class_part = 0
            else:
                class_succ = class_success_num / class_num
                class_part = partial_success_class_num/class_num
                
            result[model_name] = {"fun_success": fun_succ, "class_success": class_succ,
                                  "fun_partial_success": fun_part , "class_partial_success": class_part}

        return result

    def tear_down(self):
        # now the target directory is "classeval_evaluation"
        for module_name in list(self._imported_modules):
            try:
                if module_name in sys.modules:
                    del sys.modules[module_name]
            except KeyError:
                pass
        self._imported_modules.clear()

        p = Path.cwd().absolute()
        for cache_dir in p.glob('**/__pycache__'):
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception as e:
                print(f"Failed to remove cache directory {cache_dir}: {e}")

        gc.collect()
        reserved_files = ["evaluation.py", "path_util.py", "test_pipeline.py", "README.md", "incremental generation.png", "run.sh"]
        reserved_files += ["classeval_evaluation", "test_classes.py", "classeval_log_data.log", "results_to_tsv.py", "make_qid_test_case_jsons.py"]
        for item in p.iterdir():
            if item.name not in reserved_files and "test_pipeline" not in item.name:
                num_try = 0
                while num_try < 10:
                    try:
                        if item.is_dir():
                            for cache_dir in item.glob('**/__pycache__'):
                                shutil.rmtree(cache_dir, ignore_errors=True)
                            shutil.rmtree(item)
                        else:
                            item.unlink(missing_ok=True)
                        break
                    except:
                        print(f"err removing {item.name}, retrying")
                        num_try += 1
                        time.sleep(1)
        pycache_dirs = list(p.glob('**/__pycache__'))
        for cache_dir in pycache_dirs:
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception as e:
                print(f"Failed to remove pycache directory {cache_dir}: {e}")
        gc.collect()

