from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from jinja2 import Template

app = FastAPI()

# app.mount("/templates", StaticFiles(directory="templates"), name="templates")


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


@app.get('/', response_class=HTMLResponse)
def index():
    html = open('templates/index.html').read()
    template = Template(html)
    return template.render(title=u'Заголовок', body=u'Сообщение')


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return '/predict_item'


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return '/predict_items'
