import cProfile
import pstats
import logging
import main  # Import the other file as a module

# 로깅 설정
logging.basicConfig(filename='profiling_results.log', level=logging.INFO, format='%(message)s')

# Create a profiler object
profiler = cProfile.Profile()

# Enable the profiler
profiler.enable()

# Run the main function from the other file
main.main()

# Disable the profiler
profiler.disable()

# Create a Stats object and sort the results by cumulative time
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')

# Redirect the stats output to the logging module
stats.stream = logging.info

# Print the stats to the logging module
stats.print_stats()

print("Profiling results logged to 'profiling_results.log'")