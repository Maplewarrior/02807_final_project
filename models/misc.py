import time

def time_func(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    end = time.time()
    print(f"{func.__name__} Elapsed: {(end-start)}s")
    return out
  return wrapper

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]