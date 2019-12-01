import gc
import inspect
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.common_paths import get_data_path
from src.constants import categorical_feats
from src.general_utilities import batching

logger = logging.getLogger(__name__)


def recarray_to_array(r, dtype):
    shape = list(r.shape) + [len(r.dtype)]
    dtypes = list(zip(r.dtype.names, [dtype] * len(r.dtype.names)))
    r = r.astype(dtypes).view(dtype)
    r = r.reshape(shape)
    return np.array(r)


def shuffle_multiple(*args, axis=0):
    # Generate the permutation index array.
    permutation = np.random.permutation(args[0].shape[axis])
    # Shuffle the arrays by giving the permutation in the square brackets.
    arrays = []
    for array in args:
        arrays.append(array[permutation])
    return arrays


def check_if_integer(column, tolerance=0.01):
    """
    Checks if a column can be converted to integer
    :param column: numeric column (pd.Series)
    :param tolerance: tolerance for the float-int casting (float)
    :return: True if column can be converted to integer (bool)
    """
    casted = column.fillna(0).astype(np.int64)
    result = column - casted
    result = result.sum()
    if result > -0.01 and result < 0.01:
        return True
    else:
        return False


def reduce_mem_usage(df, int_cast=True, obj_to_category=False, subset=None):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    :param df: dataframe to reduce (pd.DataFrame)
    :param int_cast: indicate if columns should be tried to be casted to int (bool)
    :param obj_to_category: convert non-datetime related objects to category dtype (bool)
    :param subset: subset of columns to analyse (list)
    :return: dataset with the column dtypes adjusted (pd.DataFrame)
    """
    logger.info("Reducing the memory of the dataframe...")
    start_mem = df.memory_usage().sum() / 1024 ** 2
    gc.collect()
    logger.info("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    cols = subset if subset is not None else df.columns.tolist()

    for col in tqdm(cols):
        col_type = df[col].dtype

        if (
            col_type != object
            and col_type.name != "category"
            and "datetime" not in col_type.name
        ):
            c_min = df[col].min()
            c_max = df[col].max()

            # test if column can be converted to an integer
            treat_as_int = (str(col_type)[:3] == "int") and not df[col].hasnans
            if int_cast and not treat_as_int and not df[col].hasnans:
                treat_as_int = check_if_integer(df[col])

            if treat_as_int:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max
                ):
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif (
                    c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max
                ):
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif (
                    c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max
                ):
                    df[col] = df[col].astype(np.uint64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif "datetime" not in col_type.name and obj_to_category:
            df[col] = df[col].astype("category")
    gc.collect()
    end_mem = df.memory_usage().sum() / 1024 ** 2
    logger.info("Memory usage after optimization is: {:.3f} MB".format(end_mem))
    logger.info("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


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
    df1["_tmpkey"] = 1
    df2["_tmpkey"] = 1

    res = pd.merge(df1, df2, on="_tmpkey", **kwargs).drop("_tmpkey", axis=1)

    df1.drop("_tmpkey", axis=1, inplace=True)
    df2.drop("_tmpkey", axis=1, inplace=True)

    return res


def cartesian_multiple(df, columns):
    df_cartesian = df.loc[:, [columns[0]]].drop_duplicates()
    for i in range(1, len(columns)):
        df_cartesian = cartesian_pair(
            df_cartesian, df.loc[:, [columns[i]]].drop_duplicates()
        )

    return df_cartesian


def get_data_cube_from_df(df):
    shape = (
        df.store_nbr.nunique() * df.item_nbr.nunique(),
        df.date.nunique(),
        df.shape[1],
    )
    gc.collect()
    df = df.to_numpy()
    gc.collect()
    df = df.reshape(shape)
    gc.collect()
    return df


def get_records_cube_from_df(df):
    shape = (df.store_nbr.nunique() * df.item_nbr.nunique(), df.date.nunique())
    gc.collect()
    df = df.to_records(index=False)
    gc.collect()
    df = df.reshape(shape)
    gc.collect()
    return df


class FactoryLoader:
    def __init__(self):
        self.factory_dict = self.build_factory_dictionary()
        self.building_module_names = list(self.factory_dict.keys())
        self.getter = None

    def load(self, data_source_name, *args, **kwargs):
        logger.info(f"Loading {data_source_name}...")
        if not data_source_name in self.factory_dict:
            raise ValueError(
                f"Data source name provided ('{data_source_name}') has not been recognized as a valid name."
                f" Please use one of these: {', '.join(self.building_module_names)}"
            )
        else:
            self.getter = self.factory_dict[data_source_name]
            return self.getter.load(*args, **kwargs)

    def build_factory_dictionary(self):
        factory_dict = [
            (mod.data_source_name, mod()) for mod in self.get_building_modules()
        ]
        assert len(list(zip(*factory_dict))[0]) != len(
            list(set(zip(*factory_dict)))
        ), "Duplicated data source names were found, please revise the building modules definition"
        return dict(factory_dict)

    def get_building_modules(self):
        objects_in_module = list(zip(*inspect.getmembers(sys.modules[__name__])))[1]
        objects_in_module = filter(
            lambda obj: "data_source_name" in dir(obj), objects_in_module
        )
        objects_in_module = list(
            filter(lambda obj: obj.data_source_name != "__prototype", objects_in_module)
        )
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
            raise (
                FileNotFoundError(f"The requested file has not been found {filepath}")
            )
        df = pd.read_csv(filepath, **self.get_kwargs_pandas())
        return df

    def process(self, df):
        raise NotImplementedError(
            "Please, define a 'process()' method in the constructor class"
        )

    def get_data_source_name(self):
        if "data_source_name" not in dir(self.__class__):
            raise (
                NotImplementedError(
                    "Please, define a class attribute named 'data_source_name' in the constructor "
                    "indicating the name of the file being loaded"
                )
            )
        elif self.__class__.data_source_name == "__prototype":
            raise (
                NotImplementedError(
                    "Please, define a class attribute named 'data_source_name' in the constructor "
                    "indicating the name of the file being loaded"
                )
            )
        return self.__class__.data_source_name

    def get_kwargs_pandas(self):
        if "kwargs_pandas" not in dir(self.__class__):
            raise (
                NotImplementedError(
                    "Please, define a class attribute named 'kwargs_pandas' in the constructor "
                    "indicating the name of the file being loaded"
                )
            )
        return self.__class__.kwargs_pandas


class HolidaysEventsDataGetter(DataGetter):
    data_source_name = "holidays_events"
    keys = ["date"]

    def process(self, df):
        # Remove spaces and capital letters
        df[["type", "locale", "local_name"]] = df[
            ["type", "locale", "locale_name"]
        ].apply(lambda col: col.str.replace(" ", "_").str.lower())
        # Dummify the categorical vars
        # df = pd.get_dummies(df, columns=["type", "locale", "locale_name"])
        for var in ["type", "locale", "locale_name"]:
            df[var] = pd.Categorical(df[var]).codes

        # Aggregate at date level
        df = (
            df.assign(count=1)
            .drop("description", axis=1)
            .groupby("date")
            .sum()
            .astype(int)
            .rename(columns=lambda x: "holidays_" + x)  # Prefix with "holiday_"
            .reset_index()
        )

        df["holidays_count"] = (df["holidays_count"] - 2) / 2  # Center and scale
        df["holidays_transferred"] = (
            df["holidays_transferred"] - 0.5
        ) * 2  # Center and scale
        df["date"] = df.date.str.replace("-", "").astype(int)
        return df


class ItemsDataGetter(DataGetter):
    data_source_name = "items"
    keys = ["item_nbr"]

    def process(self, df):
        # TODO: Add class variable
        df = df.drop("class", axis=1)

        # Remove spaces and capital letters
        df[["family"]] = df[["family"]].apply(
            lambda col: col.str.replace(" ", "_").str.lower()
        )

        # Add the prefix "store_" to the variables
        df = df.rename(
            columns={"family": "item_family", "perishable": "item_perishable"}
        )

        # Center and scale
        df["item_perishable"] = (df["item_perishable"] - 0.5) * 2

        # Dummify the categorical vars
        # df = pd.get_dummies(df, columns=["item_family"])
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
        df[["city", "state", "type"]] = df[["city", "state", "type"]].apply(
            lambda col: col.str.replace(" ", "_").str.lower()
        )

        # Add the prefix "store_" to the variables
        df = df.rename(
            columns={
                "city": "store_city",
                "state": "store_state",
                "type": "store_type",
                "cluster": "store_cluster",
            }
        )
        # Dummify the categorical vars
        # df = pd.get_dummies(df, columns=["store_city", "store_state", "store_type", "store_cluster"])
        for var in ["store_city", "store_state", "store_type", "store_cluster"]:
            df[var] = pd.Categorical(df[var]).codes

        return df


class TestDataGetter(DataGetter):
    data_source_name = "test"
    keys = ["date", "store_nbr", "item_nbr"]

    def process(self, df):
        df["onpromotion"] = ((df["onpromotion"] - 0.5) * 2).astype(float)
        df["date"] = df.date.str.replace("-", "").astype(int)
        df["unit_sales"] = np.log1p(np.clip(df.unit_sales, 0, None))
        return df


class TrainDataGetter(DataGetter):
    data_source_name = "train"
    keys = ["date", "store_nbr", "item_nbr"]

    def process(self, df):
        df["onpromotion"] = ((df["onpromotion"] - 0.5) * 2).astype(float)
        df["date"] = df.date.str.replace("-", "").astype(int)
        df["unit_sales"] = np.log1p(np.clip(df.unit_sales, 0, None))
        return df


class TransactionsDataGetter(DataGetter):
    data_source_name = "transactions"
    keys = ["date", "store_nbr"]

    def process(self, df):
        df["date"] = df.date.str.replace("-", "").astype(int)
        df["transactions"] = np.log1p(df.transactions) - 7  # Rough center and scaling

        return df


class MasterDataGetter(DataGetter):
    data_source_name = "master"

    def load_raw(self, sample=False):
        self.fl_holidays = FactoryLoader()
        self.fl_oil = FactoryLoader()
        self.fl_transactions = FactoryLoader()
        self.fl_main = FactoryLoader()

        # Data loading
        df_holidays = self.fl_holidays.load("holidays_events")
        df_oil = self.fl_oil.load("oil")
        df_transactions = self.fl_transactions.load("transactions")
        df_main = self.fl_main.load("train")

        if sample:
            logger.info("Sampling dataframe...")
            df_main = df_main[df_main.date > int("2016-08-01".replace("-", ""))]

        logger.info("Dataset loaded successfully!")

        # Performing and merging Cartesian
        df_cartesian = cartesian_multiple(df_main, ["date", "store_nbr", "item_nbr"])
        df_cartesian = df_cartesian.sort_values(
            by=["store_nbr", "item_nbr", "date"], ascending=True
        )
        df = df_cartesian.merge(
            df_main, on=["date", "store_nbr", "item_nbr"], how="left"
        )
        del df_main, df_cartesian
        df = reduce_mem_usage(df)

        # Merge holidays
        df = df.merge(df_holidays, on=self.fl_holidays.getter.keys, how="left")
        del df_holidays
        gc.collect()
        logger.info(f"Holidays merged successfully! shape={df.shape}")

        # Merge transactions
        df = df.merge(df_transactions, on=self.fl_transactions.getter.keys, how="left")
        del df_transactions
        gc.collect()
        logger.info(f"Transactions merged successfully! shape={df.shape}")

        # Merge oil
        df = df.merge(df_oil, on=self.fl_oil.getter.keys, how="left")
        del df_oil
        gc.collect()
        logger.info(f"Oil merged successfully! shape={df.shape}")
        return df

    def process(self, df):
        df = df.fillna(0)
        logger.info(
            "NA Data filled with zeros successfully! Reducing the size of the dataset..."
        )
        df = reduce_mem_usage(df)
        logger.info("Calculating year var...")
        df["year"] = (
            df["date"].astype("str").str[0:4].astype(int) - 2015
        ) / 2  # Center and scale
        logger.info("Calculating month var...")
        df["month"] = (
            df["date"].astype("str").str[4:6].astype(int) - 6.5
        ) / 5.5  # Center and scale
        logger.info("Calculating day var...")
        df["day"] = (
            df["date"].astype("str").str[6:].astype(int) - 16
        ) / 15  # Center and scale
        logger.info("Calculating day of week var...")
        df["dayofweek"] = (
            pd.to_datetime(df["date"], format="%Y%m%d").dt.dayofweek - 3
        ) / 3  # Center and scale
        for var in categorical_feats:
            if var in df.columns:
                logger.info("Calculating {} categorical var...".format(var))
                df[var] = pd.Categorical(df[var]).codes
        df["dcoilwtico"] = (df["dcoilwtico"] - 50) / 50

        df = reduce_mem_usage(df)
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
            logger.info("Sampling dataframe...")
            df_main = df_main[df_main.date > int("2016-08-01".replace("-", ""))]

        logger.info("Dataframe loaded successfully!")

        # Performing and merging Cartesian
        df_cartesian = cartesian_multiple(df_main, ["date", "store_nbr", "item_nbr"])
        df = df_cartesian[["store_nbr", "item_nbr"]].drop_duplicates()
        df = df.sort_values(by=["store_nbr", "item_nbr"], ascending=True)
        del df_main, df_cartesian
        gc.collect()

        # Merge stores
        df = df.merge(df_stores, on=self.fl_stores.getter.keys, how="left")
        del df_stores
        gc.collect()
        logger.info("Stores merged successfully! shape={df.shape}")

        # Merge items
        df = df.merge(df_items, on=self.fl_items.getter.keys, how="left")
        del df_items
        gc.collect()
        logger.info("Items merged successfully! shape={df.shape}")
        return df

    def process(self, df):
        df = reduce_mem_usage(df)
        df = df.fillna(0)
        logger.info(
            "NA Data filled with zeros successfully! Reducing the size of the dataset..."
        )
        for var in categorical_feats:
            if var in df.columns:
                logger.info("Calculating {} categorical var...".format(var))
                df[var] = pd.Categorical(df[var]).codes
        df = reduce_mem_usage(df)
        return df


def get_categorical_cardinalities(
    data_cube, data_cube_timeless, categorical_feats, colnames, colnames_timeless
):
    categorical_cardinalities = []
    categorical_cardinalities_static = []
    for cat_var in categorical_feats:
        if cat_var in colnames:
            idx = np.where(cat_var == colnames)[0]
            categorical_cardinalities.append(int(np.max(data_cube[:, :, idx]) + 1))
        else:
            idx = np.where(cat_var == colnames_timeless)[0]
            categorical_cardinalities_static.append(
                int(np.max(data_cube_timeless[:, idx]) + 1)
            )
    return categorical_cardinalities + categorical_cardinalities_static


def get_batches_generator(
    df_time,
    df_static,
    batch_size=128,
    min_history=300,
    forecast_horizon=7,
    shuffle=True,
    shuffle_present=True,
    cuda=False,
):
    from src.constants import (
        numeric_feats,
        categorical_feats,
        target_name,
        batch_time_normalizable_feats,
        embedding_sizes,
    )

    logger.info("Shuffling dataframe...")
    df_time, df_static = shuffle_multiple(df_time, df_static)
    logger.info("Shuffle successful!")

    # Assure perfect alignment
    case_static = df_static[["store_nbr", "item_nbr"]]
    case_time = df_time[:, 0][["store_nbr", "item_nbr"]]
    assert (case_static == case_time).all()

    time_steps = df_time.shape[1]

    batcher = batching(
        list_of_iterables=[df_time, df_static],
        n=batch_size,
        return_incomplete_batches=False,
    )

    num_time_feats = np.intersect1d(numeric_feats, df_time.dtype.names)
    num_static_feats = np.intersect1d(numeric_feats, df_static.dtype.names)
    cat_time_feats = np.intersect1d(categorical_feats, df_time.dtype.names)
    cat_static_feats = np.intersect1d(categorical_feats, df_static.dtype.names)

    for batch_time, batch_static in batcher:
        if shuffle_present:
            present = random.randint(min_history, time_steps - forecast_horizon)
        else:
            present = time_steps - forecast_horizon

        # Numerical time-dependent features
        numeric_time_batch = batch_time[num_time_feats][:, :present]

        # Categorical time-dependent features
        cat_time_batch = batch_time[cat_time_feats][:, :present]

        # Numerical static features (Not defined)
        # numeric_static_batch = batch_static[num_static_feats]

        # Categorical static features
        cat_static_batch = batch_static[cat_static_feats]

        # Target
        target = batch_time[target_name][:, present : (present + forecast_horizon)]

        # Convert to arrays
        numeric_time_batch = recarray_to_array(numeric_time_batch, np.float32).swapaxes(
            0, 1
        )
        cat_time_batch = recarray_to_array(cat_time_batch, np.int32).swapaxes(0, 1)
        cat_static_batch = recarray_to_array(cat_static_batch, np.int32)
        target = target.astype(np.float32).swapaxes(0, 1)

        # Convert to torch tensors
        numeric_time_batch = torch.from_numpy(numeric_time_batch)
        cat_time_batch = torch.from_numpy(cat_time_batch).long()
        cat_static_batch = torch.from_numpy(cat_static_batch).long()
        target = torch.from_numpy(target)

        # Move to cuda if required
        if cuda:
            numeric_time_batch = numeric_time_batch.cuda()
            cat_time_batch = cat_time_batch.cuda()
            cat_static_batch = cat_static_batch.cuda()
            target = target.cuda()
        yield numeric_time_batch, cat_time_batch, cat_static_batch, target
