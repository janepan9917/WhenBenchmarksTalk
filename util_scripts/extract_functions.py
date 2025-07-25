import os
import ast
import textwrap
import json
from collections import defaultdict
from datetime import datetime
from src import DATA_DIR

def count_significant_lines(node, source):
    func_source = ast.get_source_segment(source, node)
    func_ast = ast.parse(textwrap.dedent(func_source))
    significant_lines = 0
    
    def is_docstring(node):
        return (isinstance(node, ast.Expr) and
                isinstance(node.value, ast.Str))

    for child in ast.iter_child_nodes(func_ast.body[0]):
        if is_docstring(child):
            continue
        
        child_source = ast.get_source_segment(func_source, child)
        if child_source:
            significant_lines += sum(1 for line in child_source.splitlines()
                                     if line.strip() and not line.strip().startswith('#'))

    return significant_lines

def extract_functions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    tree = ast.parse(content)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            significant_lines = count_significant_lines(node, content)
            if significant_lines > 20:
                func_body = ast.get_source_segment(content, node)
                yield node.name, textwrap.dedent(func_body), significant_lines

def get_repo_name(file_path):
    return file_path.split(os.sep)[-2]

def process_directory(directory, max_functions_per_repo=5):
    functions = defaultdict(list)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                repo_name = get_repo_name(file_path)
                if len(functions[repo_name]) >= max_functions_per_repo:
                    continue
                for func_name, func_body, significant_lines in extract_functions(file_path):
                    if len(functions[repo_name]) < max_functions_per_repo:
                        functions[repo_name].append({
                            "name": func_name,
                            "body": func_body,
                            "significant_lines": significant_lines,
                            "file_path": file_path,
                            "index": len(functions[repo_name])
                        })
                    else:
                        break
    return functions

def save_to_json(functions_by_repo, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "extracted_functions.json")
    
    new_data = []
    for repo_name, functions in functions_by_repo.items():
        for func in functions:
            new_data.append({
                "repo_name": repo_name,
                "function_name": func["name"],
                "function_body": func["body"],
                "file_path": func["file_path"],
                "index_in_repo": func["index"],
                "extraction_date": datetime.now().isoformat()
            })
    
    if os.path.exists(output_file):
        # File exists, read existing data
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # Append new data
        existing_data.extend(new_data)
        
        # Write combined data back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2)
    else:
        # File doesn't exist, create new file with new data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2)
    
    print(f"JSON data appended to: {output_file}")

def main():
    input_directory = os.path.join(DATA_DIR, "python_repos_may_2024")
    output_directory = os.path.join(DATA_DIR, "may_functions")
    
    functions_by_repo = process_directory(input_directory)
    
    total_functions = sum(len(repo_functions) for repo_functions in functions_by_repo.values())
    print(f"Found {total_functions} standalone functions (max 5 per repo) with more than 20 significant lines:")
    
    for repo_name, repo_functions in functions_by_repo.items():
        print(f"\nRepository: {repo_name}")
        for func in repo_functions:
            print(f"\n  {func['file_path']}:{func['name']}:")
            print(textwrap.indent(func['body'], '    '))
    
    save_to_json(functions_by_repo, output_directory)

if __name__ == "__main__":
    main()