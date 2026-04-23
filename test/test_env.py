from environments import CustomEnvironment

from pettingzoo.test import parallel_api_test, api_test

if __name__ == "__main__":
    env = CustomEnvironment()
    #api_test(env, num_cycles=1_000_000, verbose_progress=True)
    parallel_api_test(env, num_cycles=1_000_000)
    print("hello")