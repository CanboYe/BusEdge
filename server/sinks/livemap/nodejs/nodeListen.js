#!/usr/bin/env node

// SPDX-FileCopyrightText: 2021 Carnegie Mellon University
//
// SPDX-License-Identifier: Apache-2.0

var io = require('socket.io').listen(9000);
var io2 = require('socket.io').listen(9001);
var pg = require ('pg');

var con_string = 'postgresql://osm:'+process.env.LIVEMAP_DB_PASSWORD+'@localhost:5432/livemap_db';
console.log(con_string)
var pg_client = new pg.Client(con_string);
pg_client.connect();
var pg_client2 = new pg.Client(con_string);
pg_client2.connect();
const Query = pg.Query;


io.sockets.on('connection', function (socket) {
    socket.emit('Connected', { connected: true });
    console.log('Connected');
    socket.on('Ready for Detection Data', function (data) {
        var query = pg_client.query(new Query("SELECT row_to_json(detection) FROM detection WHERE DATE(date) = CURRENT_DATE"))
        query.on('row', (row) => {
            console.log(row);
            //var obj = JSON.parse(row_to_json);
            // console.log(row.row_to_json.image_dir);
            socket.emit('Initial', row );
        })
        query.on('end', (res) => {
            console.log('Complete');
            socket.emit('Complete', {});
        })
    });
    socket.on('Ready for More Detection Data', function (data) {
        pg_client.query('LISTEN "add_detection"');
        pg_client.on('notification', function(msg) {
            console.log("** Update Detection **")
            console.log(msg);
            socket.emit('Update', { message: msg });
        });
    });
});

io2.sockets.on('connection', function (socket2) {
    socket2.emit('Connected', { connected: true });
    console.log('Connected');
    socket2.on('Ready for Trajectory Data', function (data) {
        var query2 = pg_client2.query(new Query("SELECT row_to_json(trajectory) FROM trajectory WHERE DATE(date) = CURRENT_DATE"))
        query2.on('row', (row) => {
            console.log(row);
            //var obj = JSON.parse(row_to_json);
            //console.log(row.row_to_json.image);
            socket2.emit('Initial', row );
        })
        query2.on('end', (res) => {
            console.log('Complete');
            socket2.emit('Complete', {});
        })
    });
    socket2.on('Ready for More Trajectory Data', function (data) {
        pg_client2.query('LISTEN "add_trajectory"');
        pg_client2.on('notification', function(msg) {
            console.log("Update Trajectory")
            console.log(msg);
            socket2.emit('Update', { message: msg });
        });
    });
});
