# Interface:
#
# get_all_k_sets(n, k)
#   Returns all sets of size k from a (super)set of n elements.
#   For example, if n = 3, k = 2, returns:
#     [(0, 1), (0, 2), (1, 2)]
#
# get_all_k_permutations(n, k)
#   Returns all permutations of all sets of size k drawn from a
#     (super)set of n elements.
#   For example, if n = 3, k = 2, returns:
#     [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
#
# get_all_k_tuples(n, k)
#   Returns all possible combinations of k numbers drawn (with replacement)
#     from a set of n numbers.
#   For example, if n = 3, k = 2, returns:
#     [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
#
# increment_counter(counter, digit_limits)
#   `counter` -- a list of digits
#   `digit_limits` -- a list (same length as counter) of inclusive upper
#     limits for the digits in the counter
#   Increments the counter. For example, if counter is [0, 3, 2, 4] and limits
#     is [4, 4, 2, 4], changes counter to be [0, 4, 0, 0], then [0, 4, 0, 1],
#     etc.
#   Returns True if counter could be incremented and False if counter was at
#     maximum value.
#
# get_all_n_rankings(n)
#   Returns all possible rankings of n items where ties are allowed. To prohibit
#     ties, just use get_all_k_permutations(n, n).

# Returns False if the tuple cannot be incremented. True otherwise.
# Modifies t in place. n is the number of values a variable in the tuple
#   may take. Values are zero-indexed.
def __increment_k_set__(t, n):
    idx_to_increment = len(t) - 1
    max_idx = n - 1
    while idx_to_increment >= 0 and t[idx_to_increment] == max_idx:
        max_idx -= 1
        idx_to_increment -= 1
    if idx_to_increment < 0:
        return False
    t[idx_to_increment] += 1
    for j in range(idx_to_increment + 1, len(t)):
        t[j] = t[j - 1] + 1
    return True

def get_all_k_sets(n, k):
    current_tuple = [i for i in range(0, k)]
    stored_tuples = [tuple(current_tuple)]
    while __increment_k_set__(current_tuple, n):
        stored_tuples.append(tuple(current_tuple))
    return stored_tuples

def get_all_k_permutations(n, k):
    if n > k:
        k_sets = get_all_k_sets(n, k)
        k_permutations = get_all_k_permutations(k, k)
        results = []
        for k_set in k_sets:
            for k_perm in k_permutations:
                results.append(tuple([k_set[k_perm[i]] for i in range(0, k)]))
        return results

    # Else, n == k:
    if n == 1:
        return [tuple([0])]

    # Else, n == k > 1:
    n_minus_1_perms = get_all_k_permutations(n - 1, n - 1)
    results = []
    for perm in n_minus_1_perms:
        l = list(perm)
        for i in range(0, n):
            results.append(tuple(l[0:i] + [n - 1] + l[i:]))
    return sorted(results)

def __increment_k_tuple__(t, n):
    idx_to_increment = len(t) - 1
    while idx_to_increment >= 0 and t[idx_to_increment] == n - 1:
        idx_to_increment -= 1
    if idx_to_increment < 0:
        return False
    t[idx_to_increment] += 1
    for j in range(idx_to_increment + 1, len(t)):
        t[j] = 0
    return True

def get_all_k_tuples(n, k):
    current_tuple = [0 for i in range(0, k)]
    stored_tuples = [tuple(current_tuple)]
    while __increment_k_tuple__(current_tuple, n):
        stored_tuples.append(tuple(current_tuple))
    return stored_tuples

def increment_counter(counter, digit_limits):
    idx_to_increment = len(counter) - 1
    while idx_to_increment >= 0 and \
            counter[idx_to_increment] == digit_limits[idx_to_increment]:
        idx_to_increment -= 1
    if idx_to_increment < 0:
        return False
    counter[idx_to_increment] += 1
    for i in range(idx_to_increment + 1, len(counter)):
        counter[i] = 0
    return True

def get_all_n_rankings(n):
    rankings = __get_all_n_rankings_helper__(n)
    return sorted(rankings[-1])

def __get_all_n_rankings_helper__(n):
    if n == 1:
        return [[tuple([])], [tuple([0])]]

    the_set_of_n = set([i for i in range(0, n)])

    rankings = __get_all_n_rankings_helper__(n - 1)
    rankings.append([])
    for k in range(1, n + 1):
        k_sets = get_all_k_sets(n, k)
        for k_set in k_sets:
            k_set = set(k_set)
            for sub_ranking in rankings[n - k]:
                next_ranking = [None for i in range(0, n)]
                next_sub_idx = 0
                for idx in range(0, n):
                    if idx in k_set:
                        next_ranking[idx] = 0
                    else:
                        next_ranking[idx] = \
                            sub_ranking[next_sub_idx] + 1
                        next_sub_idx += 1
                rankings[-1].append(tuple(next_ranking))
    return rankings

if __name__ == "__main__":
    print(get_all_k_sets(3, 3))
    print(get_all_k_permutations(3, 3))
    print(get_all_k_tuples(3, 3))
    print(get_all_n_rankings(3))
