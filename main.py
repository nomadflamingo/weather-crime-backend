import uvicorn
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://weather-crime-frontend.herokuapp.com"],
)

models_stats = pd.read_csv('master_csv.csv')

states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
crime_types = ['homicide', 'violent_crime', 'robbery', 'aggravated_assault', 'property_crime', 'burglary', 'larceny', 'motor_vehicle_theft']
models = {}
for crime_type in crime_types:
    print("booting up...")
    models[crime_type] = {}

    for state in states:
        available_models = models_stats[(models_stats['Rate'] == crime_type + '_rate') & (models_stats['State'] == state)]  # find the model corresponding to crime type and state
        if len(available_models) == 0:  # if no models were found, continue
            continue

        # find best model and add prediction
        best_model = available_models[available_models['Test_R2'] == available_models['Test_R2'].max()].reset_index(drop=True).iloc[0]
        best_model_filepath = 'models/' + state + '_' + crime_type + '_' +best_model['Method'] + '.sav'
        models[crime_type][state] = {
            'R2': best_model['Test_R2'],
            'model': load(best_model_filepath)
        }



@app.get("/predict")
def predict(year: int, crime_type: str, avg_temp: float, min_temp: float, max_temp: float, pcp: float):
    # convert millimeters to inches
    pcp = pcp / 25.4
    
    results = {}
    x = pd.DataFrame([[year, avg_temp, max_temp, min_temp, pcp]], columns=['year', 'avg_temp', 'max_temp', 'min_temp', 'pcp'])

    if crime_type not in crime_types:
        raise HTTPException(status_code=400, detail="Crime type not found: " + crime_type)

    for state in states:
        if state in models[crime_type]:  # if model exists

            # get values from models
            r2 = models[crime_type][state]['R2']
            prediction = models[crime_type][state]['model'].predict(x)[0]

            # handle negative values
            if r2 < 0: r2 = 0.0001
            if prediction < 0: prediction = 0.00001
            
            # put results into dictionary
            results[state] = {
                'Accuracy': round(r2 * 100, 2),
                'Prediction': round(prediction, 5)
            }
        else:
            results[state] = None

    return results