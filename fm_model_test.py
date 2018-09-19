import unittest
from fm_model import vectorize_dic

from itertools import count
from collections import defaultdict
import numpy as np
from scipy.sparse import csr
class vectorize_dic_test(unittest.TestCase):
    def test_vectorize_dic(self):
        user_ids = ['A', 'B', 'A']
        movie_ids = ['a', 'a', 'b']
        res,_ = vectorize_dic([user_ids, movie_ids])
        print(res.todense())
        self.assertEqual(len(res.todense()), 3)