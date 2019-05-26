import os
from unittest import TestCase

from src.general_utilities import flatten, batching


class TestGeneralUtilities(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_flatten(self):
        test_list = [[1], [2, 3], [4], [5], [6]]
        self.assertListEqual([1, 2, 3, 4, 5, 6], flatten(test_list))

    def test_batching(self):
        dataset_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dataset_2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        dataset_3 = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        # Case 1
        batcher = batching(list_of_iterables=[dataset_1, dataset_2, dataset_3],
                           n=2,
                           infinite=False,
                           return_incomplete_batches=False)
        batches_1, batches_2, batches_3 = zip(*list(batcher))
        self.assertEqual(5, len(batches_1))
        self.assertEqual(5, len(batches_2))
        self.assertEqual(5, len(batches_3))
        self.assertListEqual(dataset_1, flatten(batches_1))
        self.assertListEqual(dataset_2, flatten(batches_2))
        self.assertListEqual(dataset_3, flatten(batches_3))
        # Case 2
        batcher = batching(list_of_iterables=[dataset_1, dataset_2, dataset_3],
                           n=3,
                           infinite=False,
                           return_incomplete_batches=True)
        batches_1, batches_2, batches_3 = zip(*list(batcher))
        self.assertEqual(4, len(batches_1))
        self.assertEqual(4, len(batches_2))
        self.assertEqual(4, len(batches_3))
        self.assertListEqual(dataset_1, flatten(batches_1))
        self.assertListEqual(dataset_2, flatten(batches_2))
        self.assertListEqual(dataset_3, flatten(batches_3))
        # Case 3
        batcher = batching(list_of_iterables=[dataset_1, dataset_2, dataset_3],
                           n=3,
                           infinite=False,
                           return_incomplete_batches=False)
        batches_1, batches_2, batches_3 = zip(*list(batcher))
        self.assertEqual(3, len(batches_1))
        self.assertEqual(3, len(batches_2))
        self.assertEqual(3, len(batches_3))
        self.assertListEqual(dataset_1[:-1], flatten(batches_1))
        self.assertListEqual(dataset_2[:-1], flatten(batches_2))
        self.assertListEqual(dataset_3[:-1], flatten(batches_3))
