numeric_feats = ["unit_sales", "onpromotion", "holidays_transferred", "holidays_count", "transactions",
                 "dcoilwtico", "year", "month", "day", "dayofweek"]

batch_time_normalizable_feats = ["unit_sales", "transactions", "dcoilwtico"]

categorical_feats = ["store_nbr", "item_nbr", "store_city", "store_state", "store_type", "store_cluster",
                     "holidays_type", "holidays_locale", "holidays_locale_name", "item_family"]

embedding_sizes = [100, 300, 20, 20, 20, 20, 20, 20, 20, 20]

target = "unit_sales"