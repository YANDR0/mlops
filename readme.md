# El rimi

## Training

Solo pon:
`python model_training.py`

Pero ya tenemos un modelo con < 94.46, entonces no hace falta

## API

El api se ejecuta con:

`uvicorn main:app --reload`

Puedes probar que funcione con:
* http://127.0.0.1:8000/
* http://localhost:8000

Las solicitudes post se hacen con:
* http://localhost:8000/predict

Al menos en postman, la solicitud es:

* `POST`
* A la ruta `http://localhost:8000/predict`
* Body `form-data`
* key = `file`, tipo = `file`
* value `La foto xd`

`curl -X POST -F "file=@foto.jpg" http://localhost:8000/predict`
