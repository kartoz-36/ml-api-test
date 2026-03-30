import uvicorn
from typing import Any
from fastapi import FastAPI, HTTPException
from app.model import prepare_user_sent_data, train_model
app = FastAPI()
model = None


@app.on_event("startup")
def load_model():
    global model
    model = train_model()
@app.post("/predict")
def predict(body: dict[str, Any]):
    try:
        #convert the user sent data to a dataframe same as the training data
        result = prepare_user_sent_data(body) # dont edit this line 
        """
        - predict the result
        - prediction = model.predict(result)
        - return the result in the format of "You are passed" or "You are failed"
        return {"result": youre passed if prediction[0] == 1 else "You are failed"}
 
        """
        prediction = model.predict(result)
        return {
            "result": "You are passed" if prediction[0] == 1 else "You are failed"
        }

       
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
