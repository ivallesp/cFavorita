import os
import pandas as pd
import sys
import inspect
import gc

from src.common_paths import get_data_path


def cartesian_pair(df1, df2, **kwargs):
    """
    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    See: https://github.com/pydata/pandas/issues/5401
    :param df1 dataframe 1
    :param df1 dataframe 2
    :param kwargs keyword arguments that will be passed to pd.merge()
    :return cross join of df1 and df2
    """
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res


def cartesian_multiple(df, columns):
    df_cartesian = df.loc[:, [columns[0]]].drop_duplicates()
    for i in range(1, len(columns)):
        df_cartesian = cartesian_pair(df_cartesian, df.loc[:, [columns[i]]].drop_duplicates())

    return df_cartesian


def get_data_cube_from_df(df):
    n_stores = df.store_nbr.nunique()
    n_items = df.item_nbr.nunique()
    n_timesteps = df.date.nunique()
    n_variables = df.shape[1]
    df = df.sort_values(by=["store_nbr", "item_nbr", "date"], ascending=True)
    data_cube = df.values.reshape(n_stores*n_items, n_timesteps, n_variables)
    return data_cube


class FactoryLoader:
    def __init__(self):
        self.factory_dict = self.build_factory_dictionary()
        self.building_module_names = list(self.factory_dict.keys())
        self.getter = None

    def load(self, data_source_name, *args, **kwargs):
        if not data_source_name in self.factory_dict:
            raise ValueError(f"Data source name provided ('{data_source_name}') has not been recognized as a valid name."
                             f" Please use one of these: {', '.join(self.building_module_names)}")
        else:
            self.getter = self.factory_dict[data_source_name]
            return self.getter.load(*args, **kwargs)

    def build_factory_dictionary(self):
        factory_dict = [(mod.data_source_name, mod()) for mod in self.get_building_modules()]
        assert len(list(zip(*factory_dict))[0]) != len(list(set(zip(*factory_dict)))), \
            "Duplicated data source names were found, please revise the building modules definition"
        return dict(factory_dict)

    def get_building_modules(self):
        objects_in_module = list(zip(*inspect.getmembers(sys.modules[__name__])))[1]
        objects_in_module = filter(lambda obj: "data_source_name" in dir(obj), objects_in_module)
        objects_in_module = list(filter(lambda obj:obj.data_source_name != "__prototype", objects_in_module))
        return objects_in_module


class DataGetter:  # Prototype
    data_source_name = "__prototype"
    kwargs_pandas = {"encoding": None, "sep": ","}
    keys = None

    def load(self, *args, **kwargs):
        df = self.load_raw(*args, **kwargs)
        df = self.process(df)
        return df

    def load_raw(self):
        filepath = os.path.join(get_data_path(), self.get_data_source_name() + ".csv")
        if not os.path.exists(filepath):
            raise(FileNotFoundError(f"The requested file has not been found {filepath}"))
        df = pd.read_csv(filepath, **self.get_kwargs_pandas())
        return df

    def process(self, df):
        raise NotImplementedError("Please, define a 'process()' method in the constructor class")

    def get_data_source_name(self):
        if "data_source_name" not in dir(self.__class__):
            raise(NotImplementedError("Please, define a class attribute named 'data_source_name' in the constructor "
                                      "indicating the name of the file being loaded"))
        elif self.__class__.data_source_name == "__prototype":
            raise(NotImplementedError("Please, define a class attribute named 'data_source_name' in the constructor "
                                      "indicating the name of the file being loaded"))
        return self.__class__.data_source_name

    def get_kwargs_pandas(self):
        if "kwargs_pandas" not in dir(self.__class__):
            raise(NotImplementedError("Please, define a class attribute named 'kwargs_pandas' in the constructor "
                                      "indicating the name of the file being loaded"))
        return self.__class__.kwargs_pandas


class HolidaysEventsDataGetter(DataGetter):
    data_source_name = "holidays_events"
    keys = ["date"]

    def process(self, df):
        # Remove spaces and capital letters
        df[["type", "locale", "local_name"]] = (df[["type", "locale", "locale_name"]]
                                                .apply(lambda col: col.str.replace(" ", "_").str.lower()))
        # Dummify the categorical vars
        df = pd.get_dummies(df, columns=["type", "locale", "locale_name"])

        # Aggregate at date level
        df = (df.assign(count=1)
                .drop("description", axis=1)
                .groupby("date").sum().astype(int)
                .rename(columns=lambda x: "holidays_"+x)  # Prefix with "holiday_"
                .reset_index())
        return df


class ItemsDataGetter(DataGetter):
    data_source_name = "items"
    keys = ["item_nbr"]

    def process(self, df):
        # TODO: Add class variable
        df = df.drop("class", axis=1)

        # Remove spaces and capital letters
        df[["family"]] = (df[["family"]].apply(lambda col: col.str.replace(" ", "_").str.lower()))

        # Add the prefix "store_" to the variables
        df = df.rename(columns={"family": "item_family", "perishable": "item_perishable"})

        # Dummify the categorical vars
        df = pd.get_dummies(df, columns=["item_family"])

        return df


class OilDataGetter(DataGetter):
    data_source_name = "oil"
    keys = ["date"]

    def process(self, df):
        df = df.dropna(axis=0)
        return df


class StoresDataGetter(DataGetter):
    data_source_name = "stores"
    keys = ["store_nbr"]

    def process(self, df):
        # Remove spaces and capital letters
        df[["city", "state", "type"]] = (df[["city", "state", "type"]]
                                         .apply(lambda col: col.str.replace(" ", "_").str.lower()))

        # Add the prefix "store_" to the variables
        df = df.rename(columns={"city": "store_city", "state": "store_state", "type": "store_type",
                                "cluster": "store_cluster"})
        # Dummify the categorical vars
        df = pd.get_dummies(df, columns=["store_city", "store_state", "store_type", "store_cluster"])

        return df


class TestDataGetter(DataGetter):
    data_source_name = "test"
    keys = ["date", "store_nbr", "item_nbr"]

    def process(self, df):
        return df


class TrainDataGetter(DataGetter):
    data_source_name = "train"
    keys = ["date", "store_nbr", "item_nbr"]

    def process(self, df):
        return df


class TransactionsDataGetter(DataGetter):
    data_source_name = "transactions"
    keys = ["date", "store_nbr"]

    def process(self, df):
        return df


class MasterDataGetter(DataGetter):
    data_source_name = "master"

    def load_raw(self, sample=False):
        self.fl_holidays = FactoryLoader()
        self.fl_oil = FactoryLoader()
        self.fl_transactions = FactoryLoader()
        self.fl_stores = FactoryLoader()
        self.fl_main = FactoryLoader()
        self.fl_items = FactoryLoader()

        # Data loading
        df_holidays = self.fl_holidays.load("holidays_events")
        df_oil = self.fl_oil.load("oil")
        df_transactions = self.fl_transactions.load("transactions")
        df_stores = self.fl_stores.load("stores")
        df_items = self.fl_items.load("items")
        df_main = self.fl_main.load("train")

        if sample:
            print("Sampling!")
            df_main = df_main[df_main.store_nbr < 6]

        print("Data loaded successfully!")

        # Performing and merging Cartesian
        df_cartesian = cartesian_multiple(df_main, ["date", "store_nbr", "item_nbr"])
        df = df_cartesian.merge(df_main, on=["date", "store_nbr", "item_nbr"], how="left")
        del df_main, df_cartesian

        # Merge stores
        df = df.merge(df_stores, on=self.fl_stores.getter.keys, how="left")
        del df_stores; gc.collect()
        print("Stores merged successfully!", df.shape)

        # Merge holidays
        df = df.merge(df_holidays, on=self.fl_holidays.getter.keys, how="left")
        del df_holidays; gc.collect()
        print("Holidays merged successfully!", df.shape)

        # Merge items
        df = df.merge(df_items, on=self.fl_items.getter.keys, how="left")
        del df_items; gc.collect()
        print("Items merged successfully!", df.shape)

        # Merge transactions
        df = df.merge(df_transactions, on=self.fl_transactions.getter.keys, how="left")
        del df_transactions; gc.collect()
        print("Transactions merged successfully!", df.shape)

        # Merge oil
        df = df.merge(df_oil, on=self.fl_oil.getter.keys, how="left")
        del df_oil; gc.collect()
        print("Oil merged successfully!", df.shape)
        return df

    def process(self, df):
        df["onpromotion"] = df.onpromotion.astype(float)
        df = df.fillna(0)
        df["year"] = df["date"].str[0:4].astype(int) - 2013
        df["month"] = df["date"].str[5:7].astype(int)
        df["day"] = df["date"].str[8:].astype(int)
        df["dayofweek"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.dayofweek.astype(int)
        return df
