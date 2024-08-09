import cProfile
import pstats
import io
import logging

from processor.loader import HDF5FileLoader
from config import load_config
from utils.file_util import get_run_scan_directory

config = load_config()
load_dir = config.path.load_dir
file = get_run_scan_directory(load_dir, 143, 1, 1)

logging_file = 'logs\\profiling\\profiling.log'
# 로깅 설정
logging.basicConfig(filename=logging_file, level=logging.INFO, format='%(message)s')

# Create a profiler object
profiler = cProfile.Profile()

# Enable the profiler
profiler.enable()

# Run the main function from the other file
HDF5FileLoader(file)

# Disable the profiler
profiler.disable()

# Create a Stats object and sort the results by cumulative time
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')

# Redirect the stats output to a StringIO object
output_stream = io.StringIO()
stats.stream = output_stream

# Print the stats to the StringIO object
stats.print_stats()

# Get the captured output
profiling_results = output_stream.getvalue()

# Log the captured output
logging.info(profiling_results)

print(f"Profiling results logged to '{logging_file}'")
