# Functions for reading / writing to databases

import logging
import time

import sqlalchemy
import pangres


logger = logging.getLogger(__name__)


def save_feature_importance(df):
    """Save pandas dataframe to sqlite3"""

    # Initialize sqlite database
    connection_string = "sqlite:///litmus_feature_importance.sqlite"
    db_engine = sqlalchemy.create_engine(connection_string)
    table_name = 'feature_importance'

    df['timestamp'] = time.time()
    df.set_index(["feature_names", "timestamp"], inplace=True)

    try:
        pangres.upsert(con=db_engine, df=df, table_name=table_name, if_row_exists='update',
                       chunksize=1000, create_table=True)
        logger.info(f"Successfully saved {table_name} to database")
    except Exception as e:
        logger.error(f"Error saving {table_name} to database: {e}")
