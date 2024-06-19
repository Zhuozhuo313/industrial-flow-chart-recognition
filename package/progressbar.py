import time

def progress_bar(i,start_time):
    completed = "▓" * i    
    undone = "-" * (100 - i)    
    T = time.perf_counter() - start_time
    print("\r{:^3}%[{}->{}]{:.2f}s".format(i, completed, undone, T), end="")