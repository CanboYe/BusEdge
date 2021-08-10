# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import os

import psycopg2

# connect to the testdb database
try:
    host_addr = "127.0.0.1"
    pw = os.getenv("LIVEMAP_DB_PASSWORD")
    conn = psycopg2.connect(
        database="livemap_db",
        user="osm",
        password=pw,
        host=host_addr,
        port="5432",
    )
    # create a new cursor
    cur = conn.cursor()

    # Delete data from tables and reset identity columns
    cur.execute(
        "TRUNCATE ONLY detection, trajectory, rec_images \
                 RESTART IDENTITY"
    )

    # commit the changes to the database
    conn.commit()

except (Exception, psycopg2.DatabaseError) as error:
    print(error)

finally:
    if conn is not None:
        conn.close()
