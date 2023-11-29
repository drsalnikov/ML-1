from fastapi import Body, FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from jinja2 import Template
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Annotated
import pickle


app = FastAPI()

################################# load #################################

# app.mount("/templates", StaticFiles(directory="templates"), name="templates")
with open(f"{Path(__file__).parent}\\models\\gscv_ridge", 'rb') as f:
    gscv_ridge: GridSearchCV = pickle.load(f)


with open(f"{Path(__file__).parent}\\models\\linear_reg", 'rb') as f:
    linear_reg: LinearRegression = pickle.load(f)

################################# class #################################

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


class Item_Linear(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: float
    max_power: float
    seats: float


class Items_Linear(BaseModel):
    objects: List[Item_Linear]


class Item_Ridge(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items_Ridge(BaseModel):
    objects: List[Item_Ridge]

################################# helpers #################################

def predict_linear_one(item: Item_Linear) -> float:
    list_item = [[item.year, item.km_driven,
                 item.mileage, item.engine,
                 item.max_power, item.seats]]
    colums = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
    item = pd.DataFrame(data=list_item, columns=colums)
    result = linear_reg.predict(item)
    return result


def convert(row: str):
    arr = str(row).split(' ', 1)
    if len(arr[0]) == 0:
        return float(0)
    return float(arr[0])


def prepare_convert(df, column_names, action):
    for column in column_names:
        df[column] = df[column].apply(action)


def prepare_df(df):
    colums = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
    columns_to_delete = set(df.columns.to_list())
    columns_to_delete.difference_update(set(colums))

    df.drop(columns=columns_to_delete, inplace=True)
    prepare_convert(df, ['mileage', 'engine', 'max_power'], convert)
    df[['mileage', 'engine', 'max_power', 'seats']] = (df[['mileage', 'engine', 'max_power', 'seats']].fillna(df[['mileage', 'engine', 'max_power', 'seats']].median()))
    
    print(columns_to_delete)
    df.describe(include='all')
    df.info()

    return df



################################# routes #################################

@app.get('/', response_class=HTMLResponse)
def index():
    html = open('templates/index.html').read()
    template = Template(html)
    return template.render(title=u'Homework ML-1')

    
@app.post("/predict_item")
def predict_item(item: Annotated[Item_Linear,
                                 Body(examples=[
        {
            "year": 2014,
            "km_driven": 145500,
            "mileage": 14.00,
            "engine": 2498,
            "max_power": 112.00,
            "seats": 7
        }])]) -> float:
    result = predict_linear_one(item)
    return result


@app.post("/predict_items") 
def predict_items(items: Annotated[List[Item_Linear],
                                   Body(examples=[
        [{
            "year": 2014,
            "km_driven": 145500,
            "mileage": 14.00,
            "engine": 2498,
            "max_power": 112.00,
            "seats": 7
        }, {
            "year": 2020,
            "km_driven": 100,
            "mileage": 14.00,
            "engine": 2498,
            "max_power": 112.00,
            "seats": 4
        }]])]) -> List[float]:
    result = map(predict_linear_one, items)

    return result


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)) -> List[float]:
    df = pd.read_csv(file.file)
    file.file.close()
    df_test = prepare_df(df)
    result = linear_reg.predict(df_test)
    return result
