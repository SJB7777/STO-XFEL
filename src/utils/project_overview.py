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