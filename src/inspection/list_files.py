import os

def gather_python_files(directory, output_file):
    total_length = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    outfile.write(f"===== {file_path} =====\n")
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        texts = infile.read()
                        total_length += len(texts)
                        outfile.write(texts)
                    outfile.write("\n\n")
    print(total_length)

if __name__ == "__main__":
    project_directory = '.'  # 프로젝트 디렉토리 경로를 지정하세요.
    output_file = 'project_code.txt'  # 출력 파일 이름을 지정하세요.
    gather_python_files(project_directory, output_file)