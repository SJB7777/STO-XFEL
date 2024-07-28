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

if __name__ == "__main__":
    root_directory = "./"  # Change this to your project's root directory
    exclude_files = ["config.ini"]  # List of files to exclude
    exclude_dirs = ['logs', '__pycache__']  # List of directories to exclude
    print_directory_structure(root_directory, exclude_files=exclude_files, exclude_dirs=exclude_dirs)