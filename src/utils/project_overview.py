import os
import ast

def add_parent_references(tree):
    """
    Adds parent references to each node in the AST.

    Parameters:
    - tree (ast.AST): The root node of the AST.
    """
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

def extract_functions_and_classes(file_path):
    """
    Extracts functions and classes defined in a Python file, excluding methods and names starting with '_'.

    Parameters:
    - file_path (str): The path to the Python file.

    Returns:
    - tuple: A tuple containing four lists - (global_functions, classes, class_methods).
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    tree = ast.parse(content, filename=file_path)
    add_parent_references(tree)
    
    global_functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and not node.name.startswith('_') and not isinstance(node.parent, ast.ClassDef)]
    classes = {node.name: [] for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and not node.name.startswith('_')}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.ClassDef):
            class_name = node.parent.name
            if class_name in classes:
                classes[class_name].append(node.name)
    
    return global_functions, classes

def print_directory_structure(root_dir, indent='', exclude_files=None, exclude_dirs=None):
    """
    Prints the directory structure of the given root directory, including functions and classes in Python files.

    Parameters:
    - root_dir (str): The root directory to start the traversal from.
    - indent (str): The indentation string used for formatting the output.
    - exclude_files (list): A list of file names to exclude.
    - exclude_dirs (list): A list of directory names to exclude.
    """
    if exclude_files is None:
        exclude_files = []
    if exclude_dirs is None:
        exclude_dirs = []

    try:
        items = os.listdir(root_dir)
    except PermissionError:
        print(f"{indent}[Permission Denied]")
        return

    for item in items:
        if item in exclude_files:
            continue
        
        item_path = os.path.join(root_dir, item)
        
        if os.path.isdir(item_path):
            if item in exclude_dirs:
                continue
            print(f"{indent}{item}/")
            print_directory_structure(item_path, indent + '    ', exclude_files, exclude_dirs)
        elif item.endswith('.py'):
            print(f"{indent}{item}")
            global_functions, classes = extract_functions_and_classes(item_path)
            if global_functions:
                print(f"{indent}    Functions: {', '.join(global_functions)}")
            for class_name, methods in classes.items():
                print(f"{indent}    Classes: {class_name}")
                if methods:
                    print(f"{indent}        Methods: {', '.join(methods)}")

# if __name__ == "__main__":
#     root_directory = "./"  # Change this to your project's root directory
#     exclude_files = []  # List of files to exclude
#     exclude_dirs = ['logs', '__pycache__']  # List of directories to exclude
#     print_directory_structure(root_directory, exclude_files=exclude_files, exclude_dirs=exclude_dirs)

import os
import ast
import importlib
import inspect
from collections import defaultdict

def analyze_project(root_dir):
    project_structure = defaultdict(lambda: {"files": [], "modules": {}})
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == ".":
            rel_path = ""
        
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                module_name = os.path.splitext(filename)[0]
                
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                
                tree = ast.parse(content)
                classes, functions = [], []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                
                loc = len(content.splitlines())
                
                project_structure[rel_path]["files"].append(filename)
                project_structure[rel_path]["modules"][module_name] = {
                    "classes": classes,
                    "functions": functions,
                    "loc": loc
                }
    
    return project_structure

import os
import ast
from collections import defaultdict

def analyze_project(root_dir):
    project_structure = defaultdict(lambda: {"files": [], "modules": {}})
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == ".":
            rel_path = ""
        
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                module_name = os.path.splitext(filename)[0]
                
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                
                tree = ast.parse(content)
                classes, functions = [], []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                
                loc = len(content.splitlines())
                
                project_structure[rel_path]["files"].append(filename)
                project_structure[rel_path]["modules"][module_name] = {
                    "classes": classes,
                    "functions": functions,
                    "loc": loc
                }
    
    return project_structure

def print_project_structure(structure, indent=""):
    for dir_name, dir_content in structure.items():
        print(f"{indent}{dir_name}/")
        for file in dir_content["files"]:
            print(f"{indent}  {file}")
        
        for module_name, module_info in dir_content["modules"].items():
            print(f"{indent}  {module_name}.py:")
            print(f"{indent}    Classes: {', '.join(module_info['classes'])}")
            print(f"{indent}    Functions: {', '.join(module_info['functions'])}")
            print(f"{indent}    Lines of Code: {module_info['loc']}")
        
        # 하위 디렉토리 처리를 위한 재귀 호출 제거

if __name__ == "__main__":
    root_dir = "."  # 프로젝트 루트 디렉토리 경로
    project_structure = analyze_project(root_dir)
    
    print("Project Structure:")
    print_project_structure(project_structure)