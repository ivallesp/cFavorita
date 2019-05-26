import unittest
from src.data_tools import FactoryLoader


class TestDataFactory(unittest.TestCase):
    def test_items_table_shape(self):
        fl = FactoryLoader()
        df = fl.load("items")
        self.assertEqual((4100, 3), df.shape)

    def test_holidays_events_table_shape(self):
        fl = FactoryLoader()
        df = fl.load("holidays_events")
        self.assertEqual((312, 6), df.shape)

    def test_oil_table_shape(self):
        fl = FactoryLoader()
        df = fl.load("oil")
        self.assertEqual((1175, 2), df.shape)

    def test_stores_table_shape(self):
        fl = FactoryLoader()
        df = fl.load("stores")
        self.assertEqual((54, 5), df.shape)

    def test_test_table_shape(self):
        fl = FactoryLoader()
        df = fl.load("test")
        self.assertEqual((3370464, 5), df.shape)

    def test_train_table_shape(self):
        fl = FactoryLoader()
        df = fl.load("train")
        self.assertEqual((125497040, 6), df.shape)

    def test_transactions_table_shape(self):
        fl = FactoryLoader()
        df = fl.load("transactions")
        self.assertEqual((83488, 3), df.shape)

    def test_items_table_primary_keys(self):
        fl = FactoryLoader()
        df = fl.load("items")
        self.assertTrue(df.item_nbr.is_unique)
        self.assertFalse(df.item_nbr.hasnans)

    def test_holidays_events_table_primary_keys(self):
        fl = FactoryLoader()
        df = fl.load("holidays_events")
        self.assertTrue(df.date.is_unique)
        self.assertFalse(df.date.hasnans)

    def test_oil_table_primary_keys(self):
        fl = FactoryLoader()
        df = fl.load("oil")
        self.assertTrue(df.date.is_unique)
        self.assertFalse(df.date.hasnans)

    def test_stores_table_primary_keys(self):
        fl = FactoryLoader()
        df = fl.load("stores")
        self.assertTrue(df.store_nbr.is_unique)
        self.assertFalse(df.store_nbr.hasnans)

    def test_test_table_primary_keys(self):
        fl = FactoryLoader()
        df = fl.load("test")
        self.assertTrue(df.id.is_unique)
        self.assertFalse(df.id.hasnans)
        self.assertEqual(df[["date", "store_nbr", "item_nbr"]].drop_duplicates().shape[0], df.shape[0])
        self.assertFalse(df.date.hasnans)
        self.assertFalse(df.store_nbr.hasnans)
        self.assertFalse(df.item_nbr.hasnans)

    def test_train_table_primary_keys(self):
        fl = FactoryLoader()
        df = fl.load("train")
        self.assertEqual((125497040, 6), df.shape)
        self.assertTrue(df.id.is_unique)
        self.assertFalse(df.id.hasnans)
        self.assertEqual(df[["date", "store_nbr", "item_nbr"]].drop_duplicates().shape[0], df.shape[0])
        self.assertFalse(df.date.hasnans)
        self.assertFalse(df.store_nbr.hasnans)
        self.assertFalse(df.item_nbr.hasnans)

    def test_transactions_table_primary_keys(self):
        fl = FactoryLoader()
        df = fl.load("transactions")
        self.assertEqual(df[["date", "store_nbr"]].drop_duplicates().shape[0], df.shape[0])
        self.assertFalse(df.date.hasnans)
        self.assertFalse(df.store_nbr.hasnans)

    def test_incorrect_name(self):
        incorrect_ref = "holidays"
        self.assertRaises(ValueError, lambda: FactoryLoader().load(incorrect_ref))

    def test_prototype_name(self):
        incorrect_ref = "__prototype"
        self.assertRaises(ValueError, lambda: FactoryLoader().load(incorrect_ref))
