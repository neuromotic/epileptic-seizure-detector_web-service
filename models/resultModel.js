const mongo = require("mongoose");

const resultScheme = new mongo.Schema({
    dateId: { type: String, required: true },
    result: { type: String },
});

module.exports = mongo.model("Result", resultScheme);