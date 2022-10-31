# weather-crime-backend

To run the app, go to the project folder and run
```
python main.py
```
Note: it might take a while to start, because it's reading all the prediction models, but the requests should run quickly


In case you see a warning "Trying to unpickle estimator DecisionTreeRegressor from version 1.0.2 when using version 1.1.2.", it's probably better to run
```
pip install scikit-learn==1.0.2
```
although seems to work without it

You can see the endpoint info at **localhost:8000/docs**

The example output should look like this:
```
{
    "AK": {
        "R2": 0.9619922929697092,
        "Prediction": 12.840342548789335
    },
    "AL": {
        "R2": 0.3246551139900667,
        "Prediction": 12.534189521247908
    },
    "AR": {
        "R2": 0.0635085228658955,
        "Prediction": 11.498736807712461
    },
    "AZ": null,
    ...
}
```
null indicates that no models were given for the current state (probably because some data is missing)
