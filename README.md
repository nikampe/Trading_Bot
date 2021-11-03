# Trading_Bot
Trading Bot using Chart Pattern & Time Series Analysis

## Setup
1. MySQL setup on your local device
2. MongoDB Atlas setup
3. MySQL user setup for apprpriate user rights to read and write
4. MongoDB user setup for apprpriate user rights to read and write
5. In db.py: Insert the HOST, USER, PASSWORD and DB_NAME to connect to your MySQL database
6. In db.py: Insert the CONNECTION_STRING to connect to your MongoDB Atlas Cloud database
7. Run db.py in order to create the tables in your MySQL database and the collections in your MondoDB Atlas database
8. Kraken account setup
9. Creation of API credentials in the Kraken user settings
10. In krakenapi.py: Insert the API_KEY_PUBL and API_KEY_PRIV

## Run
1. Activate the virtual environment: source VIRTUAL_ENV/bin/activate 
2. Run RUN.py

## Upcoming projects
1. Cloud deployment on AWS with 5min run schedule
2. Sophisticated NLP Sentiment Analysis based on current lexion approach
3. Reinforcement learning and other approaches for portfolio weight optimization 
4. Portfolio dashboard & anaylsis with data visualizations using Dash and fundamental KPIs
   ...
5. Adaptation to the stock market with the top 50 S&P500 stocks
