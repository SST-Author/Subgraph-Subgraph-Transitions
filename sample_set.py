import random

# A set that allows all operations, _including_ random sampling, to be performed
#   in O(1) time.
class SampleSet:

    def __init__(self):
        self.__list_version__ = []
        self.__indices__ = {}

    def add(self, elt):
        if elt not in self.__indices__:
            self.__indices__[elt] = len(self.__list_version__)
            self.__list_version__.append(elt)

    def remove(self, elt):
        if elt not in self.__indices__:
            raise KeyError("Error! element: %s not found in SampleSet." % elt)
        if self.__indices__[elt] == len(self.__list_version__) - 1:
            self.__list_version__.pop()
            del self.__indices__[elt]
        else:
            gap_filler = self.__list_version__.pop()
            self.__list_version__[self.__indices__[elt]] = gap_filler
            self.__indices__[gap_filler] = self.__indices__[elt]
            del self.__indices__[elt]

    def randomly_sample(self):
        return random.choice(self.__list_version__)

    def __iter__(self):
        return self.__list_version__.__iter__()

    def __contains__(self, elt):
        return elt in self.__indices__

    def __len__(self):
        return len(self.__list_version__)

    def __str__(self):
        return "SampleSet(%s)" % str(self.__list_version__)

if __name__ == "__main__":
    s = SampleSet()
    s.add(1)
    s.add(3)
    s.add(2)
    s.add(4)
    s.remove(3)
    print(list(s) == [1, 4, 2])
    s.remove(4)
    print(list(s) == [1, 2])
    print(s.randomly_sample() in s)
    s.add(4)
    print(not 3 in s)
    print(4 in s)
    s.add(3)
    print(list(s) == [1, 2, 4, 3])
    print(len(s) == 4)
