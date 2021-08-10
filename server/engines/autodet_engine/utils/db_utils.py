# SPDX-FileCopyrightText: 2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import math
import os

import numpy as np
import psycopg2

logger = logging.getLogger(__name__)

PW = os.getenv("LIVEMAP_DB_PASSWORD", "my_password")
IN_DOCKER = os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False)
if IN_DOCKER:
    HOST_ADDR = "172.17.0.1"
else:
    HOST_ADDR = "127.0.0.1"


class DB_Manager:
    def __init__(self):
        # connect to the PostgreSQL database
        # conn = psycopg2.connect(**params)
        self.conn = psycopg2.connect(
            database="livemap_db",
            user="osm",
            password=PW,
            host=HOST_ADDR,
            port="5432",
        )
        # create a new cursor
        self.cur = self.conn.cursor()

    def __del__(self):
        self.cur.close()
        self.conn.close()

    def select_GPS(self, lat_min, lat_max, long_min, long_max, cls):
        SQL = "SELECT * FROM detection WHERE (latitude BETWEEN {} AND {}) \
               AND (longitude BETWEEN {} AND {}) AND (type = '{}')".format(
            lat_min, lat_max, long_min, long_max, cls
        )

        try:
            self.cur.execute(SQL)
            records = self.cur.fetchall()
            if len(records) != 0:
                logger.info("repeated records!")
                # for row in records:
                #     print(row)
                return False
            else:
                return True

        except (Exception, psycopg2.DatabaseError) as error:
            logger.error(error)

    def insert_rec_images(self, lat, long, alt, img_dir, bbox, cam_id, timestamp):
        """insert multiple vendors into the vendors table"""

        SQL = "INSERT INTO rec_images (id_number, camera_id, timestamp, date, latitude, longitude, altitude, bounding_box, image_dir) \
               VALUES (DEFAULT, %s, %s, %s, %s, %s, %s, %s, %s)"

        try:
            # execute the INSERT statement
            # cur.executemany(sql,vendor_list)
            # imageLocation = 's_frame'+str(self.predCounter)+'.jpg'
            self.cur.execute(
                SQL,
                (
                    cam_id,
                    timestamp,
                    datetime.date.today(),
                    lat,
                    alt,
                    long,
                    bbox,
                    img_dir,
                ),
            )
            # commit the changes to the database
            self.conn.commit()
            logger.debug("inserted row to TABLE rec_images")

        except (Exception, psycopg2.DatabaseError) as error:
            logger.error(error)

    def insert_detection(
        self, lat, long, alt, img_id, img_dir, bbox, cls, cam_id, timestamp
    ):
        """insert multiple vendors into the vendors table"""

        SQL = "INSERT INTO detection (id_number, detection_id, camera_id, timestamp,  date, type, latitude, longitude, altitude, bounding_box, image_dir) \
               VALUES (DEFAULT, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

        try:
            # execute the INSERT statement
            # cur.executemany(sql,vendor_list)
            # imageLocation = 's_frame'+str(self.predCounter)+'.jpg'
            self.cur.execute(
                SQL,
                (
                    img_id,
                    cam_id,
                    timestamp,
                    datetime.date.today(),
                    cls,
                    lat,
                    long,
                    alt,
                    [int(bbox[i]) for i in range(4)],
                    img_dir,
                ),
            )
            # commit the changes to the database
            self.conn.commit()
            logger.debug("inserted row to TABLE detection")

        except (Exception, psycopg2.DatabaseError) as error:
            logger.error(error)

    def select_and_insert(
        self,
        lat,
        long,
        alt,
        img_id,
        img_dir,
        bbox,
        cls,
        cam_id,
        timestamp,
        dist_thres=2,
    ):
        lat_thres, long_thres = self.gps2threshold(dist_thres, lat)
        if self.select_GPS(
            lat - lat_thres, lat + lat_thres, long - long_thres, long + long_thres, cls
        ):
            self.insert_detection(
                lat, long, alt, img_id, img_dir, bbox, cls, cam_id, timestamp
            )
            # print(lat, long)
            return True
        else:
            return False

    def insert_trajectory(self, lat, long, alt):
        """insert multiple vendors into the vendors table"""

        sql = "INSERT INTO trajectory (id_number, bus_id, date, latitude, longitude, altitude, velocity, heading)\
               VALUES (DEFAULT, %s, %s, %s, %s, %s, NULL, NULL)"

        try:
            # execute the INSERT statement
            self.cur.execute(sql, ("1", datetime.date.today(), lat, long, alt))
            # commit the changes to the database
            self.conn.commit()
            logger.debug("inserted row to TABLE trajectory")
        except (Exception, psycopg2.DatabaseError) as error:
            logger.error(error)

    def gps2threshold(self, dist_meter, lat):
        r = 6378137  # meters
        C_long = 2 * r * math.pi
        C_lat = C_long * math.cos(lat)
        lat_thres = dist_meter * 360 / C_long
        long_thres = dist_meter * 360 / C_lat
        return abs(lat_thres), abs(long_thres)
