<!--
SPDX-FileCopyrightText: 2021 Carnegie Mellon University

SPDX-License-Identifier: Apache-2.0
-->

# Livemap sink

## Set up LiveMap server

1. [Install Node.js](https://github.com/nodesource/distributions/blob/master/README.md#debinstall)
2. [Set Up OpenStreetMap Tile Server](https://www.linuxbabe.com/ubuntu/openstreetmap-tile-server-ubuntu-18-04-osm)
3. Set up database for LiveMap:
    - Set a passwork for user osm

        ```sh
        sudo -u postgres psql
        \password osm
        ```

    - Create a new database for LiveMap

        ```sh
        sudo -u osm createdb livemap_db
        ```

    - Set the password as an environment variable:

        ```sh
        export LIVEMAP_DB_PASSWORD={your_password}
        ```

        ```sh
        # Or add it to .bashrc (recommended)
        echo "export LIVEMAP_DB_PASSWORD={your_password}" >> ~/.bashrc
        source ~/.bashrc
        ```

    - Create tables and functions for LiveMap

        ```sh
        python3 utils/create_database.py
        ```

4. Set up web server:
    - Copy contents of html folder into /var/www/html
    - Create a symlink to the folder to save the image results:

        ```sh
        ln -s path_to_folder/images /var/www/html/images/cloudletImages
        ```

5. If we want to reset the database: `python3 utils/reset_db.py`

## Launch LiveMap

1. Run SQL listener: `./nodejs/nodeListen.js`
2. Open LiveMap webpage: `http://deluge.elijah.cs.cmu.edu/index.html`
   (or your own web server.)
