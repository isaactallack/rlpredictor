# Rocket League Predictor
Predicts likelihood of winning a Rocket League game from game statistics - see if you are unlucky or lucky!

Tensorflow prediction model to assess team performance in a game.

Only works on 2v2 games currently and some player ID data is hardcoded in the create_df.py file - this is to filter out unwanted replays.

Uses 110 statistics from ballchasing.com, such as:
- shots for/against
- % in zones (halves, thirds) for/against
- bpm for/against
- boost for/against
- demos for/against

'main.py' - takes a URL from ballchasing.com and downloads all replays on the page currently displayed. Make sure to delete all files from the 'csvs' folder before using this otherwise they will not be saved!
'create_df' - merges and filters all replay statistic files creating 'combined_data.csv', 'filtered_data.csv', 'raw_train.csv' and 'train.csv'. Only 'train.csv' is of interest. This contains all feature points + the result (for training/validation) and the goal difference (this is not used and discarded but was included for some testing of loss functions).
'model.py' - this is the Tensorflow training model and create the model for predictions.
'predict.py' - use this to predict the results from 'train.csv'.
