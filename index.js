const express = require('express');
const bodyParser = require("body-parser");
const mongo = require('./db/conectionMongo');
const fileUpload = require('express-fileupload');
const app = express();


/* MIDDLE | bodyParser for parse to JSON format */
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(fileUpload());

/* IMPORT ROUTER MODULES */
app.get("/", (req, res) => {
    res.send({
        description: 'Access denied',
        response: []
    });
});

const routesApp = require("./routes/appRoutes")(app);

mongo.conectar(app);