import os
import pandas as pd
import sys
import inspect
import gc
import random

import numpy as np
from src.common_paths import get_data_path
from src.general_utilities import batching
from src.constants import numeric_feats, categorical_feats, target, batch_time_normalizable_feats, embedding_sizes


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
        #df = pd.get_dummies(df, columns=["type", "locale", "locale_name"])
        for var in ["type", "locale", "locale_name"]:
            df[var] = pd.Categorical(df[var]).codes

        # Aggregate at date level
        df = (df.assign(count=1)
                .drop("description", axis=1)
                .groupby("date").sum().astype(int)
                .rename(columns=lambda x: "holidays_"+x)  # Prefix with "holiday_"
                .reset_index())

        df["holidays_count"] = (df["holidays_count"] - 2) / 2  # Center and scale
        df["holidays_transferred"] = (df["holidays_transferred"] - 0.5) * 2  # Center and scale
        df["date"] = df.date.str.replace("-", "").astype(int)
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

        # Center and scale
        df["item_perishable"] = (df["item_perishable"] - 0.5) * 2

        # Dummify the categorical vars
        #df = pd.get_dummies(df, columns=["item_family"])
        for var in ["item_family"]:
            df[var] = pd.Categorical(df[var]).codes

        return df


class OilDataGetter(DataGetter):
    data_source_name = "oil"
    keys = ["date"]

    def process(self, df):
        df = df.dropna(axis=0)
        df["date"] = df.date.str.replace("-", "").astype(int)
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
        #df = pd.get_dummies(df, columns=["store_city", "store_state", "store_type", "store_cluster"])
        for var in ["store_city", "store_state", "store_type", "store_cluster"]:
            df[var] = pd.Categorical(df[var]).codes


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
        df["onpromotion"] = ((df["onpromotion"] - 0.5) * 2).astype(float)
        df["date"] = df.date.str.replace("-", "").astype(int)
        return df


class TransactionsDataGetter(DataGetter):
    data_source_name = "transactions"
    keys = ["date", "store_nbr"]

    def process(self, df):
        df["date"] = df.date.str.replace("-", "").astype(int)
        return df


class MasterDataGetter(DataGetter):
    data_source_name = "master"

    def load_raw(self, sample=False):
        self.fl_holidays = FactoryLoader()
        self.fl_oil = FactoryLoader()
        self.fl_transactions = FactoryLoader()
        # self.fl_stores = FactoryLoader()
        self.fl_main = FactoryLoader()
        # self.fl_items = FactoryLoader()

        # Data loading
        df_holidays = self.fl_holidays.load("holidays_events")
        df_oil = self.fl_oil.load("oil")
        df_transactions = self.fl_transactions.load("transactions")
        # df_stores = self.fl_stores.load("stores")
        # df_items = self.fl_items.load("items")
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
        # df = df.merge(df_stores, on=self.fl_stores.getter.keys, how="left")
        # del df_stores; gc.collect()
        # print("Stores merged successfully!", df.shape)

        # Merge holidays
        df = df.merge(df_holidays, on=self.fl_holidays.getter.keys, how="left")
        del df_holidays; gc.collect()
        print("Holidays merged successfully!", df.shape)

        # Merge items
        # df = df.merge(df_items, on=self.fl_items.getter.keys, how="left")
        # del df_items; gc.collect()
        # print("Items merged successfully!", df.shape)

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
        df = df.fillna(0)
        df, _ = reduce_mem_usage(df)
        df["onpromotion"] = df.onpromotion.astype(float)
        df["year"] = (df["date"].astype("str").str[0:4].astype(int) - 2015)/2  # Center and scale
        df["month"] = (df["date"].astype("str").str[5:7].astype(int)-6.5)/5.5  # Center and scale
        df["day"] = (df["date"].astype("str").str[8:].astype(int)-16)/15  # Center and scale
        df["dayofweek"] = (pd.to_datetime(df["date"], format="%Y%m%d").dt.dayofweek-3)/3  # Center and scale
        for var in ["store_nbr", "item_nbr"]:
            df[var] = pd.Categorical(df[var]).codes
        return df


class MasterTimelessGetter(DataGetter):
    data_source_name = "master_timeless"

    def load_raw(self, sample=False):
        self.fl_stores = FactoryLoader()
        self.fl_main = FactoryLoader()
        self.fl_items = FactoryLoader()

        # Data loading
        df_stores = self.fl_stores.load("stores")
        df_items = self.fl_items.load("items")
        df_main = self.fl_main.load("train")

        if sample:
            print("Sampling!")
            df_main = df_main[df_main.store_nbr < 6]

        print("Data loaded successfully!")

        # Performing and merging Cartesian
        df_cartesian = cartesian_multiple(df_main, ["date", "store_nbr", "item_nbr"])
        df = df_cartesian[["store_nbr", "item_nbr"]].drop_duplicates()
        del df_main, df_cartesian; gc.collect()

        # Merge stores
        df = df.merge(df_stores, on=self.fl_stores.getter.keys, how="left")
        del df_stores; gc.collect()
        print("Stores merged successfully!", df.shape)

        # Merge items
        df = df.merge(df_items, on=self.fl_items.getter.keys, how="left")
        del df_items; gc.collect()
        print("Items merged successfully!", df.shape)

    def process(self, df):
        df = df.fillna(0)
        df = reduce_mem_usage(df)
        return df


def get_categorical_cardinalities(data_cube, categorical_feats, colnames):
    categorical_feats_idx = np.where(np.expand_dims(categorical_feats, 0) == np.expand_dims(colnames, 1))[0]
    categorical_cardinalities = []
    for cat_var in categorical_feats_idx:
        categorical_cardinalities.append(int(np.max(data_cube[:, :, cat_var])))
    return categorical_cardinalities


def get_batcher_generator(data_cube, model, batch_size, colnames, shuffle_every_epoch=True,
                          history_window_size=1000, prediction_window_size=30):
    numeric_feats_idx = np.where(np.expand_dims(numeric_feats, 0) == np.expand_dims(colnames, 1))[0]
    categorical_feats_idx = np.where(np.expand_dims(categorical_feats, 0) == np.expand_dims(colnames, 1))[0]
    batch_time_normalizable_feats_idx = np.where(np.expand_dims(batch_time_normalizable_feats, 0) == np.expand_dims(colnames, 1))[0]
    target_idx = np.where(target == colnames)[0]
    numeric_feats_norm_idx = \
    np.where(np.expand_dims(batch_time_normalizable_feats_idx, 0) == np.expand_dims(numeric_feats_idx, 1))[0]

    if shuffle_every_epoch:
        np.random.shuffle(data_cube)

    time_axis_size = data_cube.shape[1]

    for batch in batching(list_of_iterables=data_cube, n=batch_size, return_incomplete_batches=False):
        assert (len(batch) == 1 and type(batch) == list)
        batch = batch[0]
        t0 = random.randint(0, time_axis_size-prediction_window_size-history_window_size)
        t1 = t0 + history_window_size
        t2 = t1 + prediction_window_size
        numeric_batch = batch[:, t0:t1, numeric_feats_idx].astype(float)
        categorical_batch = batch[:, t0:t1, categorical_feats_idx].astype(int)
        target_batch = batch[:, t1:t2, target_idx].astype(float)

        # In batch normalization
        target_mean = batch[:, t0:t1, target_idx].astype(float).mean(axis=1, keepdims=True)
        target_std = batch[:, t0:t1, target_idx].astype(float).std(axis=1, keepdims=True)

        target_std[target_std == 0] = 1  # Avoid infinite

        target_batch = (target_batch-target_mean)/target_std

        # Calculate the indices over the numeric batch
        for feat_idx in numeric_feats_norm_idx:
            feat_mean = numeric_batch[:, :, [feat_idx]].astype(float).mean(axis=1, keepdims=True)
            feat_std = numeric_batch[:, :, [feat_idx]].astype(float).std(axis=1, keepdims=True)
            feat_std[feat_std == 0] = 1  # Avoid infinite
            numeric_batch[:, :, [feat_idx]] = (numeric_batch[:, :, [feat_idx]] - feat_mean)/feat_std

        # Prepare tensorflow batch
        tf_batch = dict()
        for cat_i in range(categorical_batch.shape[2]):
            tf_batch[getattr(model.ph, "cat_{}".format(cat_i))] = categorical_batch[:, :, [cat_i]]

        tf_batch[model.ph.numerical_feats] = numeric_batch
        tf_batch[model.ph.target] = target_batch

        yield tf_batch


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")
        # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist
