# Functions for reading / writing to databases

import logging

import pangres
import sqlalchemy


logger = logging.getLogger(__name__)


def save_df_to_db(df, table_name):
    """Save pandas dataframe to sqlite3"""

    # Initialize sqlite database
    connection_string = "sqlite:///litmus.sqlite"
    db_engine = sqlalchemy.create_engine(connection_string)

    try:
        pangres.upsert(con=db_engine, df=df, table_name=table_name, if_row_exists="update",
                       chunksize=1000, create_table=True)
        logger.info(f"Successfully saved {table_name} to database")
    except Exception as e:
        logger.error(f"Error saving {table_name} to database: {e}")
