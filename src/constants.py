numeric_feats = ["unit_sales",
                 "onpromotion",
                 "holidays_transferred",
                 "holidays_count",
                 "transactions",
                 "dcoilwtico",
                 "year",
                 "month",
                 "day",
                 "dayofweek"]

batch_time_normalizable_feats = ["unit_sales",
                                 "transactions",
                                 "dcoilwtico"]

categorical_feats = ["store_nbr",
                     "item_nbr",
                     "store_city",
                     "store_state",
                     "store_type",
                     "store_cluster",
                     "holidays_type",
                     "holidays_locale",
                     "holidays_locale_name",
                     "item_family"]

embedding_sizes = {"store_nbr": 100,
                   "item_nbr": 300,
                   "store_city": 20,
                   "store_state": 20,
                   "store_type": 20,
                   "store_cluster": 20,
                   "holidays_type": 20,
                   "holidays_locale": 20,
                   "holidays_locale_name": 20,
                   "item_family": 20}

target = "unit_sales"
