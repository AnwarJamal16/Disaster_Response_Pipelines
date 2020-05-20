# Disaster Response Pipeline Project


### Overview:
In a natural disaster situation people communicates on social media expressing their needs and asks for help. We want to keep tracks of those relevant conversation, identifying people who are in need by categorizing the message and text messages sent around that location at that time. For example, we would like to know if a message is related to the disaster or not, or if the message is about food shortage, water depletion, or a child being alone. The results may then be forwarded to disaster relief agencies so that help can be provided promptly.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



###Files/Folders :
All the below folders are contained in disaster_response_pipeline_project.

1. app: contains all the files relating to the web API development.

2. data: contains the raw data, data processing script and its corresponding jupyter notebook.

3. models: contains the model training and saving script along with its jupyter notebook.

4. imgs: contains screenshots of tha app.

'Note: Look into each folder's README to find more information about its contents.'


### Requirements :

Flask

NLTK

numpy

pandas

plotly

sklearn

sqlalchemy
