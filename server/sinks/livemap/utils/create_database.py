# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import os

import psycopg2

# Create tables
CREATE_RECEIVED_IMAGES_TABLE = "CREATE TABLE rec_images(\
                       id_number                SERIAL PRIMARY KEY      NOT NULL, \
                       camera_id                TEXT    NOT NULL, \
                       timestamp                TEXT    NOT NULL, \
                       date                     DATE     NOT NULL, \
                       latitude                 DOUBLE PRECISION        NOT NULL, \
                       longitude                DOUBLE PRECISION        NOT NULL, \
                       altitude                 DOUBLE PRECISION    NOT NULL, \
                       bounding_box             INTEGER[], \
                       image_dir                    TEXT   NOT NULL);"

CREATE_DETECTION_TABLE = "CREATE TABLE detection(\
                       id_number                SERIAL PRIMARY KEY      NOT NULL, \
                       detection_id             TEXT    NOT NULL, \
                       camera_id                TEXT    NOT NULL, \
                       timestamp                TEXT    NOT NULL, \
                       date                     DATE     NOT NULL, \
                       type                     TEXT    NOT NULL, \
                       latitude                 DOUBLE PRECISION        NOT NULL, \
                       longitude                DOUBLE PRECISION        NOT NULL, \
                       altitude                 DOUBLE PRECISION    NOT NULL, \
                       bounding_box             INTEGER[4], \
                       image_dir                    TEXT     NOT NULL);"

CREATE_TRAJECTORY_TABLE = "CREATE TABLE trajectory(\
                      id_number                SERIAL PRIMARY KEY      NOT NULL, \
                      bus_id                   TEXT    NOT NULL, \
                      date                     DATE               NOT NULL, \
                      latitude                 DOUBLE PRECISION    NOT NULL, \
                      longitude                DOUBLE PRECISION    NOT NULL, \
                      altitude                 DOUBLE PRECISION    NOT NULL, \
                      velocity                 DOUBLE PRECISION, \
                      heading                  DOUBLE PRECISION);"


# Create notify and trigger
CREATE_NOTIFY_DETECTION = "CREATE OR REPLACE FUNCTION notify_detection() RETURNS trigger\n \
                        LANGUAGE plpgsql\n \
                        AS $$\n \
                        BEGIN\n \
                            PERFORM pg_notify('add_detection', json_build_object('table', TG_TABLE_NAME, 'detection_id', NEW.detection_id, \
                            'type', NEW.type, 'latitude', NEW.latitude, 'longitude', NEW.longitude, 'image_dir', NEW.image_dir)::text);\n \
                            RETURN NULL;\n \
                        END;\n \
                        $$;"
SQL_TRIGGER_DETECTION = "DROP TRIGGER IF EXISTS update_detection_notify ON detection;\n \
                       CREATE TRIGGER update_detection_notify AFTER INSERT ON detection \
                       FOR EACH ROW EXECUTE PROCEDURE notify_detection();"

CREATE_NOTIFY_TRAJECTORY = "CREATE OR REPLACE FUNCTION notify_trajectory() RETURNS trigger\n \
                    LANGUAGE plpgsql\n \
                    AS $$\n \
                    BEGIN\n \
                        PERFORM pg_notify('add_trajectory', json_build_object('table', TG_TABLE_NAME, 'latitude', NEW.latitude, \
                        'longitude', NEW.longitude)::text);\n \
                        RETURN NULL;\n \
                    END;\n \
                    $$;"

SQL_TRIGGER_TRAJECTORY = "DROP TRIGGER IF EXISTS update_trajectory_notify ON trajectory;\n \
                     CREATE TRIGGER update_trajectory_notify AFTER INSERT ON trajectory \
                     FOR EACH ROW EXECUTE PROCEDURE notify_trajectory();"

# connect to the your_dbname database
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
    # Drop tables
    cur.execute("DROP TABLE IF EXISTS rec_images")
    cur.execute("DROP TABLE IF EXISTS detection")
    cur.execute("DROP TABLE IF EXISTS trajectory")

    cur.execute(CREATE_RECEIVED_IMAGES_TABLE)
    cur.execute(CREATE_DETECTION_TABLE)
    cur.execute(CREATE_TRAJECTORY_TABLE)

    cur.execute(CREATE_NOTIFY_DETECTION)
    cur.execute(SQL_TRIGGER_DETECTION)

    cur.execute(CREATE_NOTIFY_TRAJECTORY)
    cur.execute(SQL_TRIGGER_TRAJECTORY)
    # commit the changes to the database
    conn.commit()

except (Exception, psycopg2.DatabaseError) as error:
    print(error)

finally:
    if conn is not None:
        conn.close()
