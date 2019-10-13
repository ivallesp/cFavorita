import boto3
import tempfile

import pandas as pd

from psycopg2 import connect
from src.config import load_config
from src.exceptions import CredentialsError


def upload_dataframe_to_s3(df, bucket_name, output_filename):
    """Uploads a dataframe to s3 in csv format.

    Args:
        df (pd.DataFrame): table intended to be uploaded.
        bucket_name (str): name of the bucket.
        output_filename (str): name that the file will take in s3.
    """
    temporary_filepath = tempfile.mkstemp()[1]
    df.to_csv(temporary_filepath, sep=",", index=False)
    upload_file_to_s3(
        bucket_name=bucket_name,
        input_filepath=temporary_filepath,
        output_filename=output_filename,
    )


def upload_file_to_s3(bucket_name, input_filepath, output_filename):
    """Takes a file existing locally and uploads it to the desired S3 bucket.

    Args:
        bucket_name (str): name of the bucket.
        input_filepath (str): path of an existing file.
        output_filename (str): name that the file will take in s3.
    """
    s3 = boto3.client("s3")
    with open(input_filepath, "rb") as f:
        s3.upload_fileobj(f, bucket_name, output_filename)


def list_files_in_s3_bucket(bucket_name):
    """Lists the content of a given s3 bucket.

    Args:
        bucket_name (str): name of an existing bucket in the configured aws account

    Returns:
        list: contents of the s3 bucket
    """
    s3 = boto3.client("s3")
    bucket = s3.list_objects(Bucket=bucket_name)
    contents = [c["Key"] for c in bucket["Contents"]]
    return contents


def delete_file_from_s3(bucket_name, filepath):
    """Deletes an existing file in the specified bucket. If the file does not exist,
    raises an error.

    Args:
        bucket_name (str): name of an existing bucket in the configured aws account.
        filepath (str): remote path in s3 bucket to the file intended to be removed.

    Raises:
        ValueError: if the file specified is not found in the s3 bucket
    """
    if filepath not in list_files_in_s3_bucket(bucket_name):
        raise FileNotFoundError(
            f"The filepath specified '{filepath}' does not exist in the"
            f" bucket '{bucket_name}'"
        )
    s3 = boto3.client("s3")
    s3.delete_object(Bucket=bucket_name, Key=filepath)


def copy_csv_from_s3_to_db(bucket_name, filepath, destination_table, db_name):
    """Copies an existing csv file from s3 to a specified DB.

    Args:
        bucket_name (str): name of an existing bucket in the configured aws account.
        filepath (str): remote path in s3 bucket to the file intended to be unloaded.
        destination_table (str): structure.table to use to unload the csv content.
        db_name (str): name of the db where the table will be unloaded.
    """
    if filepath not in list_files_in_s3_bucket(bucket_name):
        raise FileNotFoundError(
            f"The filepath specified '{filepath}' does not exist in the"
            f" bucket '{bucket_name}'"
        )
    creds = boto3.Session().get_credentials()
    query = (
        f"COPY {destination_table} from 's3://{bucket_name}/{filepath}' credentials "
        f"'aws_access_key_id={creds.access_key};"
        f"aws_secret_access_key={creds.secret_key}' "
        f"ignoreheader 1 removequotes delimiter ',' region 'us-east-1'"
    )
    config = load_config("credentials")
    if db_name not in config.keys():
        raise CredentialsError(f'Credentials "{db_name}" not found')
    credentials = {k: v for k, v in config.items() if type(v) is not dict}
    credentials = {**credentials, **config[db_name]}

    with connect(**credentials) as conn:
        conn.cursor().execute(query)
        conn.commit()


if __name__ == "__main__":
    # Example of usage; to be removed.
    bucket_name = "test655321"
    filename = "test99.csv"
    example_df = pd.util.testing.makeDataFrame()
    upload_dataframe_to_s3(
        df=example_df, bucket_name=bucket_name, output_filename=filename
    )
    db_name = "eusopdw"
    destination_table = "stif_output.test"

    list_of_files = list_files_in_s3_bucket(bucket_name)

    if filename in list_of_files:
        copy_csv_from_s3_to_db(
            bucket_name=bucket_name,
            filepath=filename,
            destination_table=destination_table,
            db_name=db_name,
        )
        delete_file_from_s3(bucket_name=bucket_name, filepath=filename)
