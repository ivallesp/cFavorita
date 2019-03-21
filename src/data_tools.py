import os
import pandas as pd
import sys
import inspect

from src.common_paths import get_data_path


class DataGetter:  # Prototype
    data_source_name = "__prototype"
    kwargs_pandas = {"encoding": None, "sep": ","}

    def load(self):
        df = self.load_raw()
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
        elif self.__class__.data_source_name == "prototype":
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

    def process(self, df):
        df = df.assign(holiday=1)
        return df


class ItemsDataGetter(DataGetter):
    data_source_name = "items"

    def process(self, df):
        return df


class OilDataGetter(DataGetter):
    data_source_name = "oil"

    def process(self, df):
        df = df.dropna(axis=0)
        return df


class FactoryLoader:
    def __init__(self):
        self.factory_dict = self.build_factory_dictionary()
        self.building_module_names = list(self.factory_dict.keys())

    def load(self, data_source_name):
        if not data_source_name in self.factory_dict:
            raise ValueError(f"Data source name provided ('{data_source_name}') has not been recognized as a valid name."
                             f" Please use one of these: {', '.join(self.building_module_names)}")
        else:
            return self.factory_dict[data_source_name].load()

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
