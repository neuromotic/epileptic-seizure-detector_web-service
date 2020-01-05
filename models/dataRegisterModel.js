const mongo = require("mongoose");

const dataRegisterScheme = new mongo.Schema({
    name: { type: String, required: true },
    comment: { type: String },
    date: { type: String }
});

module.exports = mongo.model("DataRegister", dataRegisterScheme);