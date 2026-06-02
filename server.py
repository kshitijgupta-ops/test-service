from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from TrueFoundry!"}

@app.get("/predict")
def predict():
    # Add your model inference logic here
    return {"prediction": "result"}