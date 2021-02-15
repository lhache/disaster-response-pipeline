# Disaster Response Pipeline Project

This project uses processed data from <a href="https://www.figure-eight.com/dataset/combined-disaster-response-data/">Figure Eight's Disaster Response data</a>.
It uses a multilabel classifier to classify any message as one or more disaster related categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python3 app/run.py`

3. Go to http://0.0.0.0:3001/


### Files
- /app/run.py - run the web app
- /data - data files and processing
    - process_data.py
    - disaster_messages.csv
    - disaster_categories.csv
- /models/train_classifier.py - train the classifier and save the model
