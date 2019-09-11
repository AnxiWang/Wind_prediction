from multiprocessing import Pool
import os


def f(x):
    print('process id:', os.getpid())
    print(x)
    return x*x


if __name__ == '__main__':
    with Pool(8) as p:
        print(p.map(f, range(200)))
