const mongo = require("mongoose");

mongo.Promise = global.Promise;

module.exports = {
    conectar: async app => {
        await mongo.connect("mongodb+srv://neuromotic_mobile_app:n3ur04pp@cluster0-xtptv.mongodb.net/detectorEventosEpileptiformes", {
            useNewUrlParser: true
        });

        app.listen(5000, () => {
            console.log("MongoDB and server connected");
        });
    }
};