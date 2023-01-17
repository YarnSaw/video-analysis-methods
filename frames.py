import os, json, shutil, csv, re, cv2


directory = 'subset/data'
def leastmostframes(directory):
    longest = 0
    shortest = 10000000
    #Iterate through every file
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        #read video
        video = cv2.VideoCapture(f)
        #print(video.get(7))
        curFrames = video.get(7)

        if longest < curFrames:
            longest = curFrames
        
        if shortest > curFrames:
            shortest = curFrames

        


    return shortest, longest

print(leastmostframes(directory))