import cProfile
import pstats
import io
import logging
import main  # Import the other file as a module
from core.loader_strategy import HDF5FileLoader

file = "D:\\dev\\xfel_sample_data\\run=001\scan=001\p0110.h5"
def read_hdf5(file: str):
    return HDF5FileLoader(file)

logging_file = 'logs\\profiling\\profiling.log'
# 로깅 설정
logging.basicConfig(filename=logging_file, level=logging.INFO, format='%(message)s')

# Create a profiler object
profiler = cProfile.Profile()

# Enable the profiler
profiler.enable()

# Run the main function from the other file
# main.main()
read_hdf5(file)

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