import os, json, shutil, csv, re, cv2


def findIDsForTags(tags, fileName):
  labels = []
  with open(f'labels/{fileName}') as file:
    labels = json.load(file)
  
  ids = []
  rows = []
  for label in labels:
      if label['template'] in tags:
        video = cv2.VideoCapture(f"20bn-something-something-v2/{label['id']}.webm")
        if video.get(3) == 427 and video.get(4) == 240:
          ids.append(label['id'])
          rows.append(label)

  with open (f'subset-{fileName}', 'w+') as file:
    json.dump(rows, file, indent=1)
  return ids

def copyIDsToNewDirectory(directory, ids):
  try:
    os.mkdir(directory)
  except:
    pass # directory already exists
  for id in ids:
    shutil.copy(f'20bn-something-something-v2/{id}.webm', f'{directory}/{id}.webm')


if __name__ == "__main__":
  tags = ['Pushing [something] from left to right', 'Putting [something] on a surface', 'Uncovering [something]']
  direct = "subset2"
  file = 'train.json'
  ids = findIDsForTags(tags, file)
  copyIDsToNewDirectory(direct, ids)

  file = 'validation.json'
  ids = findIDsForTags(tags, file)
  copyIDsToNewDirectory(direct, ids)


  # the test set uses a csv instead of json for the file that contains the answers
  reader = None
  ids = [];
  testJson = [];
  newCSV = [];
  with open('labels/test-answers.csv') as file:
    reader = csv.reader(file, delimiter=";")
    for row in reader:
      if re.sub(r'something', r'[something]', row[1]) in tags: # this one needs the square braces around something so I don't need to redefine tags - being lazy = more work
        ids.append(row[0])
        testJson.append({"id":f"{row[0]}"})
        newCSV.append(row);
  copyIDsToNewDirectory(direct, ids)
  with open ('subset-test.json', 'w+') as file:
    json.dump(testJson, file, indent=1)
  with open ('subset-test-answers.csv', 'w+') as file:
    writer = csv.writer(file, delimiter=';')
    for row in newCSV:
      writer.writerow(row);