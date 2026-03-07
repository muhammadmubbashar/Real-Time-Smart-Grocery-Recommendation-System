from collections import namedtuple
DataIngestionConfig = namedtuple("DataIngestionConfig", ["download_dataset_url", "raw_data_dir", "ingested_dir"])