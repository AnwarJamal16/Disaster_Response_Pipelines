# Disaster Response Pipeline Project

> ### Overview:
In a natural disaster situation people communicate through different means expresses their needs and asks for help. We want to keep track of those relevant messages, identifying people who are in need of categorizing the message. We are using disaster data from Figure Eight to build a model that classifies disaster messages. We will develop a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

> ### Requirements :

* Flask

* plotly

* NLTK

* sklearn

* sqlalchemy

> ### Files/Folders Structure:

* app

  - template

    - master.html   []: # main page of web app

    - go.html  []: # classification result page of web app

* data

  - disaster_categories.csv  []: # data to process 

  - disaster_messages.csv  []: # data to process

  - process_data.py

  - InsertDatabaseName.db   []: # database to save clean data to

* models

 - train_classifier.py

 - classifier.pkl  []: # saved model 

All the above folders are contained in Disaster_Response_Pipelines.

1. app: contains all the files relating to the web API development.

2. data: contains the raw data, data processing script and its corresponding jupyter notebook.

3. models: contains the model training and saving script along with its jupyter notebook.

4. imgs: contains screenshots of tha app.

> ### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



> ### Screenshots :

Here are some screeshots of the API:

![Alt text](/path/to/image.jpg)