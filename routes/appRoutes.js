const appController = require('../controllers/appController');

module.exports = app => {
    /* ROUTES */
    /* Get Result */
    app.get('/getResult', appController.getResults);
    /* UploadFile */
    app.post('/uploadFile', appController.uploadFile);
    /* UploadData */
    app.post('/uploadData', appController.uploadData);
};