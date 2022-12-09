import pandas as pd
import os
import numpy as np
import warnings
import json

os.chdir(f'{os.getcwd()}/training')

warnings.filterwarnings("ignore",category=FutureWarning)

# Create an empty dataframe to store the combined data
df = pd.DataFrame()
bad_groups = []

# Loop through the files in the 'csvs' folder
for file in os.listdir('csvs'):
  # Read the data from the file into a dataframe
  file_df = pd.read_csv(f'csvs/{file}', sep=';')

  # Add a new column to the dataframe with the file name
  file_df['file_name'] = file

  # Append the dataframe to the combined dataframe
  #df = df.append(file_df, ignore_index=True)
  df = pd.concat([df, file_df])

# Save the combined dataframe to a file
df.to_csv('combined_data.csv', index=False)

# Read the data from the 'combined_data.csv' file
df = pd.read_csv('combined_data.csv')

# Group the data by the 'file_name' column and get the size of each group
group_sizes = df.groupby('file_name')['file_name'].size()

# Filter out the groups that don't have exactly 4 records
df = df[df['file_name'].isin(group_sizes[group_sizes == 4].index)]

groups = df.groupby('file_name')
for group in groups:
    if '76561198845707031' and '76561198080302493' not in pd.DataFrame(group[1])['player id'].values:
        bad_groups += [group[0]]

for drp_grp in bad_groups:
    df = df[df['file_name'] != drp_grp]
    
# Save the updated data to a new file
df.to_csv('filtered_data.csv', index=False)

# Read the data from the 'filtered_data.csv' file
df = pd.read_csv('filtered_data.csv')

################################
# Now aggregate stats as required

# Load the aggregation operations from the 'operations.json' file
with open('team1.json') as f:
  team1_ops = json.load(f)

with open('team2.json') as f:
  team2_ops = json.load(f)

# Define a function that creates two lines for a group
def create_lines(group):
  group = pd.DataFrame(group[1])
  # Create a new dataframe with the same columns as the group
  lines = pd.DataFrame(columns=group.columns)

  # Add the first team to the new dataframe
  if '76561198845707031' and '76561198080302493' in group['player id'].values:
    lines = lines.append(group.agg(team1_ops), ignore_index=True)
    lines['team-name'] = 'a'
  else:
    lines = lines.append(group.agg(team2_ops), ignore_index=True)
    lines['team-name'] = 'b'


  return lines

# Read the data from the 'games.csv' file
df = pd.read_csv('filtered_data.csv')
df_teams = pd.DataFrame()

# Group the data by the 'file_name' and 'color' columns
groups = df.groupby(['file_name', 'color'])
for group in groups:
    df_teams = pd.concat([df_teams, create_lines(group)])

# Save the updated data to a new file
df_teams.to_csv('teams.csv', index=False)


#########################################


# Read the data from the 'games.csv' file
df = pd.read_csv('teams.csv')

#df = df.dropna(axis=1, how='all')

# Select the rows for teams 'a' and 'b'
team_a = df[df['team-name'] == 'a'].dropna(axis=1, how='all')
team_b = df[df['team-name'] == 'b'].dropna(axis=1, how='all')

# Create a new dataframe with the combined data
combined = pd.DataFrame()
x = pd.DataFrame()
y = pd.DataFrame()

# Add the columns from team 'a' with the '-a' suffix
for column in team_a.columns:
  x[column + '-a'] = team_a[column]

# Add the columns from team 'b' with the '-b' suffix
for column in team_b.columns:
  y[column + '-b'] = team_b[column]

x, y = x.reset_index(), y.reset_index()

combined = pd.concat([x, y], axis = 1, join = 'inner')

# Save the combined data to a new file
combined.to_csv('raw_train.csv', index=False)

###########################################

df = pd.read_csv('raw_train.csv')

# Define a custom function that returns 1 if the 'score-a' column
# is greater than the 'score-b' column, and 0 otherwise
def result(row):
  if row['goals-a'] > row['goals conceded-a']:
    return 1
  else:
    return 0

# Define a custom function that returns 1 if the 'score-a' column
# is greater than the 'score-b' column, and 0 otherwise
def gd(row):
  if row['goals-a'] > row['goals conceded-a']:
    return 1
  else:
    return 0

# Apply the custom function to each row in the dataframe
# and save the result in a new 'result' column
df['result'] = df.apply(result, axis=1)

df['gd'] = df['goals-a']-df['goals conceded-a']

# Read the column names to drop from the 'columns_to_drop.json' file
with open('columns_to_drop.json') as f:
  columns_to_drop = json.load(f)

# Drop the columns from the dataframe using the names in the 'columns_to_drop' list
df = df.drop(columns_to_drop['columns'], axis = 1)

df.to_csv('train.csv', index=False)