import numpy

def select_max_cap(k=5):
    list_k = [i / k for i in range(1, k + 1)]
    max_cap = numpy.random.choice(list_k)
    possible_p_list = list_k[:list_k.index(max_cap)+1]
    return max_cap, possible_p_list


if __name__ == "__main__":
    ml = [1,2,3,4,5]
    print(ml[:0+1])