import os
import pandas as pd
import sys
import inspect
import gc
import random

from tqdm import tqdm
import numpy as np
from src.common_paths import get_data_path
from src.general_utilities import batching
from src.constants import numeric_feats, categorical_feats, target, batch_time_normalizable_feats, embedding_sizes


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
    result = (column - casted)
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
    start_mem = df.memory_usage().sum() / 1024 ** 2;
    gc.collect()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    cols = subset if subset is not None else df.columns.tolist()

    for col in tqdm(cols):
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()

            # test if column can be converted to an integer
            treat_as_int = str(col_type)[:3] == 'int'
            if int_cast and not treat_as_int:
                treat_as_int = check_if_integer(df[col])

            if treat_as_int:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name and obj_to_category:
            df[col] = df[col].astype('category')
    gc.collect()
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.3f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

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
    shape = df.store_nbr.nunique() * df.item_nbr.nunique(), df.date.nunique(), df.shape[1]
    gc.collect()
    df = df.to_numpy()
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
        self.fl_main = FactoryLoader()

        # Data loading
        df_holidays = self.fl_holidays.load("holidays_events")
        df_oil = self.fl_oil.load("oil")
        df_transactions = self.fl_transactions.load("transactions")
        df_main = self.fl_main.load("train")

        if sample:
            print("Sampling!")
            df_main = df_main[df_main.store_nbr < 6]

        print("Data loaded successfully!")

        # Performing and merging Cartesian
        df_cartesian = cartesian_multiple(df_main, ["date", "store_nbr", "item_nbr"])
        df_cartesian = df_cartesian.sort_values(by=["store_nbr", "item_nbr", "date"], ascending=True)
        df = df_cartesian.merge(df_main, on=["date", "store_nbr", "item_nbr"], how="left")
        del df_main, df_cartesian

        # Merge holidays
        df = df.merge(df_holidays, on=self.fl_holidays.getter.keys, how="left")
        del df_holidays; gc.collect()
        print("Holidays merged successfully!", df.shape)

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
        print("NA Data filled with zeros successfully! Reducing the size of the dataset...")
        df = reduce_mem_usage(df)
        print("Calculating year var...")
        df["year"] = (df["date"].astype("str").str[0:4].astype(int) - 2015)/2  # Center and scale
        print("Calculating month var...")
        df["month"] = (df["date"].astype("str").str[4:6].astype(int)-6.5)/5.5  # Center and scale
        print("Calculating day var...")
        df["day"] = (df["date"].astype("str").str[6:].astype(int)-16)/15  # Center and scale
        print("Calculating day of week var...")
        df["dayofweek"] = (pd.to_datetime(df["date"], format="%Y%m%d").dt.dayofweek-3)/3  # Center and scale
        for var in categorical_feats:
            if var in df.columns:
                print("Calculating {} categorical var...".format(var))
                df[var] = pd.Categorical(df[var]).codes
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
            print("Sampling!")
            df_main = df_main[df_main.store_nbr < 6]

        print("Data loaded successfully!")

        # Performing and merging Cartesian
        df_cartesian = cartesian_multiple(df_main, ["date", "store_nbr", "item_nbr"])
        df = df_cartesian[["store_nbr", "item_nbr"]].drop_duplicates()
        df = df.sort_values(by=["store_nbr", "item_nbr"], ascending=True)
        del df_main, df_cartesian; gc.collect()

        # Merge stores
        df = df.merge(df_stores, on=self.fl_stores.getter.keys, how="left")
        del df_stores; gc.collect()
        print("Stores merged successfully!", df.shape)

        # Merge items
        df = df.merge(df_items, on=self.fl_items.getter.keys, how="left")
        del df_items; gc.collect()
        print("Items merged successfully!", df.shape)
        return df

    def process(self, df):
        df = df.fillna(0)
        print("NA Data filled with zeros successfully! Reducing the size of the dataset...")
        for var in categorical_feats:
            if var in df.columns:
                print("Calculating {} categorical var...".format(var))
                df[var] = pd.Categorical(df[var]).codes
        df = reduce_mem_usage(df)
        return df


def get_categorical_cardinalities(data_cube, data_cube_timeless, categorical_feats, colnames, colnames_timeless):
    categorical_cardinalities = []
    categorical_cardinalities_static = []
    for cat_var in categorical_feats:
        if cat_var in colnames:
            idx = np.where(cat_var == colnames)[0]
            categorical_cardinalities.append(int(np.max(data_cube[:, :, idx])+1))
        else:
            idx = np.where(cat_var == colnames_timeless)[0]
            categorical_cardinalities_static.append(int(np.max(data_cube_timeless[:, idx]) + 1))
    return categorical_cardinalities+categorical_cardinalities_static


def get_batcher_generator(data_cube_time, data_cube_timeless, model, batch_size, colnames_time, colnames_timeless,
                          history_window_size=1000, prediction_window_size=30):
    from src.constants import numeric_feats, categorical_feats, target, batch_time_normalizable_feats, embedding_sizes
    numeric_feats = np.array(numeric_feats)
    categorical_feats = np.array(categorical_feats)
    batch_time_normalizable_feats = np.array(batch_time_normalizable_feats)
    colnames_time = np.array(colnames_time)
    colnames_timeless = np.array(colnames_timeless)

    num_idx_time = np.where(numeric_feats[None] == np.expand_dims(colnames_time, 1))[0]
    cat_idx_time = np.where(np.setdiff1d(categorical_feats, ["store_nbr", "item_nbr"])[None] == colnames_time[:, None])[0]
    batch_time_normalizable_feats_idx = np.where(batch_time_normalizable_feats[None] == colnames_time[:, None])[0]
    target_idx = np.where(target == colnames_time)[0]
    numeric_feats_norm_idx = np.where(batch_time_normalizable_feats_idx[None] == num_idx_time[:,None])[0]
    num_idx_timeless = np.where(numeric_feats[None] == colnames_timeless[:, None])[0]
    cat_idx_timeless = np.where(categorical_feats[None] == colnames_timeless[:, None])[0]

    categorical_feats_in_batch = [colnames_time[i] for i in cat_idx_time] + [colnames_timeless[i] for i in cat_idx_timeless]

    time_axis_size = data_cube_time.shape[1]
    batcher =  batching(list_of_iterables=[data_cube_time, data_cube_timeless], n=batch_size,
                                          return_incomplete_batches=False)


    for batch, batch_timeless in batcher:
        t0 = random.randint(0, time_axis_size-prediction_window_size-history_window_size)
        t1 = t0 + history_window_size
        t2 = t1 + prediction_window_size
        numeric_batch = batch[:, t0:t1, num_idx_time].astype(float)
        categorical_batch = batch[:, t0:t1, cat_idx_time].astype(int)
        target_batch = batch[:, t1:t2, target_idx].astype(float)

        categorical_batch_static = batch_timeless[:, cat_idx_timeless].astype(int)[:, None, :] * \
                                   np.ones([1, categorical_batch.shape[1], 1])

        categorical_batch = np.concatenate([categorical_batch, categorical_batch_static], axis=-1)

        numeric_batch_static = batch_timeless[:, num_idx_timeless].astype(float)[:, None, :] * \
                                   np.ones([1, categorical_batch.shape[1], 1])

        numeric_batch = np.concatenate([numeric_batch, numeric_batch_static], axis=-1)

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
        for cat_i, cat_var in enumerate(categorical_feats_in_batch):
            tf_batch[getattr(model.ph, "cat_{}".format(cat_var))] = categorical_batch[:, :, [cat_i]]

        tf_batch[model.ph.numerical_feats] = numeric_batch
        tf_batch[model.ph.target] = target_batch

        yield tf_batch, (target_mean, target_std)
