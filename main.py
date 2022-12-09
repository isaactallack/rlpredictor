import csv
import os
import urllib.request
import time

os.chdir(f'{os.getcwd()}/training')

# Open the input file and read the URLs
with open('urls.csv', 'r') as input_file:
  reader = csv.reader(input_file)
  urls = [row[0] for row in reader]

# Find the index of the last downloaded file
last_index = 0
for file in os.listdir('csvs'):
  if file.startswith('csv_'):
    index = int(file[4:-4])
    if index > last_index:
      last_index = index

# Download and save the remaining URLs to files in the 'csvs' folder
for i, url in enumerate(urls[last_index+1:]):
  response = urllib.request.urlopen(url)
  data = response.read()

  # Save the file as 'csvs/csv_<i>.csv'
  with open('csvs/csv_{i+last_index+1}.csv', 'wb') as output_file:
    output_file.write(data)

  time.sleep(5)