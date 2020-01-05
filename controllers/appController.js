const path = require('path');
const { spawn } = require('child_process');
const dataRegister = require('../models/dataRegisterModel');
const resultData = require('../models/resultModel');

module.exports = {

    uploadFile: async (req, res) => {
        try {
            console.log('uploadfile');
            let nameFile = 'file.csv';
            let { date } = req.body;
            let indexResult = 0;
            if (Object.keys(req.files).length == 0) {
                return res.status(400).send({
                    code: 400,
                    description: 'No files were uploaded.',
                    response: null
                });
            }
            let e4File = req.files.e4File;
            e4File.mv(`../DetectorEventosEpileptiformes_Server/controllers/${nameFile}`, function (err) {
                const path = `../DetectorEventosEpileptiformes_Server/controllers/${nameFile}`;
                if (err)
                    return res.status(500).send(err);

                const subprocess = runScript();

                subprocess.stdout.on('data', (data) => {
                    console.log((`${data}`).toString());
                    pushData(date, (`${data}`).toString());
                });
                subprocess.stderr.on('data', (data) => {
                    console.log(`error:${data}`);
                });

                res.status(200).send({
                    code: 200,
                    description: 'success',
                    response: path
                });
            });
        } catch (error) {
            console.log('uploat request error', error); // the uploaded file object
            res.status(500).send({
                code: 500,
                description: 'Server error',
                message: error
            });
        }
    },

    /* Upload Data */
    uploadData: async (req, res) => {
        console.log('request uploadData');
        try {
            let register = req.body;
            dataRegister.create(register, function (err, response) {
                const dataResult = response;
                res.send({
                    code: 200,
                    description: 'success',
                    response: dataResult
                });
            });
        } catch (error) {
            console.log('Error uploadData: ' + error);
            res.status(500).send({
                code: 500,
                description: 'Server error',
                message: error
            });
        }
    },
    /* GET RESULT */
    getResults: async (req, res) => {
        try {
            console.log('getResults request');
            const dateId = req.query.dateId;
            const result = await resultData.findOne({ dateId: dateId });
            if (result) {
                res.send({
                    code: 200,
                    description: 'success',
                    response: result
                });
            } else {
                res.send({
                    code: 404,
                    description: 'not found',
                    response: {}
                });
            }
        } catch (error) {
            res.status(500).send({
                code: 500,
                description: 'server error',
                message: error
            });
        }
    },
}


function runScript() {
    return spawn('python', [
        path.join(__dirname, 'preprocessing_classification.py')
    ]);
}

function pushData(date, result) {
    console.log('OK');
    try {
        const object = {
            dateId: date,
            result: result
        }
        resultData.create(object, function (err, response) {
            const dataResult = response;
            console.log('RESPONSE STORED DATA', dataResult);
        });
    } catch (error) {
        console.log('Error upload result: ' + error);
    }
}
