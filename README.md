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
