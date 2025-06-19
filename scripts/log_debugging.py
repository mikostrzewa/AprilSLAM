import re
import os

def parse_and_print_log(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            for line_number, line in enumerate(file, 1):
                # Regular expression to match the log format and extract the message
                match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+)', line.strip())
                if match:
                    timestamp, level, message = match.groups()
                    # Print in a more readable format
                    print(f"[{timestamp}] {level}: {message}")
                else:
                    # If the line doesn't match the expected format, print it as is
                    print(f"Line {line_number}: {line.strip()}")
    except FileNotFoundError:
        print(f"File {log_file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Get the project root directory
    script_dir = os.path.dirname(__file__)
    project_root = os.path.join(script_dir, '..')
    log_file_path = os.path.join(project_root, 'data', 'logs', 'last_run.log')
    
    parse_and_print_log(log_file_path)