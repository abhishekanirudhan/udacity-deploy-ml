# Machine Learning & CI/CD with Github, S3 and Heroku

> repository: https://github.com/abhishekanirudhan/udacity-deploy-ml

Welcome to a simple model developed as a part of Udacity's Machine Learning DevOps Nanodegree. This was my first attempt at writing up my own API (FastAPI), implementing CI (Github Actions), and deploying a trained model (S3 and Heroku). The trained model is available at: https://udacity-ml-app.herokuapp.com/.

## Querying
You can query using CURL, try:

```
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{
        "age": 31, 
        "workclass": "Private",
        "education": "Masters",
        "marital-status": "Never-married",
        "occupation":"Prof-speciality",
        "relationship":"Not-in-family",
        "race":"White",
        "sex":"Female",
        "capital-gain":14000,
        "capital-loss":0,
        "hours-per-week":55,
        "native-country":"United-States"
    }' \
  https://udacity-ml-app.herokuapp.com/predict

```
Alternatively, try the `requests` module via `python`

```
python sample_api_requests.py
```