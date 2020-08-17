from os import walk
from os.path import join
import pickle as pkl

for subdir, dirs, files in walk("test_results/"):
    for filename in files:
        if "_ssts_" not in filename:
            continue
        full_file_path = join(subdir, filename)
        f = open(full_file_path, "rb")
        values = pkl.load(f)
        f.close()
        if len(values) > 10000:
            print(full_file_path)
            print(len(values))
            new_values = values[0:10000]

            f = open(full_file_path, "wb")
            pkl.dump(new_values, f)
            f.close()
