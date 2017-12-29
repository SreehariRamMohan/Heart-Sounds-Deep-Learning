import time
from functools import wraps


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              ("time", str(t1 - t0))
              )
        return result

    return function_timer

@fn_timer
def function_that_I_want_to_time():
    sum = 0
    for i in range(1, 1000):
        sum += i
    print(sum)


print("To check the time it takes a function to execute")
function_that_I_want_to_time()

print("To check the time for lines of code to run:")
import time
start = time.clock()
sum = 0
for i in range(1, 1000):
    sum += i
print(sum)
print(time.clock() - start)
