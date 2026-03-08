from collections import namedtuple
DataIngestionConfig = namedtuple("DataIngestionConfig", ["download_dataset_url", "raw_data_dir", "ingested_dir"])
DataValidationConfig = namedtuple("DataValidationConfig", ["order_csv_file", "product_csv_file", "order_products_prior_csv_file", "order_products_train_csv_file", "department_csv_file", "aisles_csv_file", "clean_data_dir", "serialized_objects_dir"]) 