# USAGE
# python parse_xml.py --input ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml --output ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train_eyes.xml
# python parse_xml.py --input ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml --output ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test_eyes.xml

# import the necessary packages
import argparse
import re

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to iBug 300-W data split XML file")
ap.add_argument("-t", "--output", required=True,
	help="path output data split XML file")
args = vars(ap.parse_args())

# in the iBUG 300-W dataset, each (x, y)-coordinate maps to a specific
# facial feature (i.e., eye, mouth, nose, etc.) -- in order to train a
# dlib shape predictor on *just* the eyes, we must first define the
# integer indexes that belong to the eyes
LANDMARKS = set(list(range(36, 48)))

# to easily parse out the eye locations from the XML file we can
# utilize regular expressions to determine if there is a 'part'
# element on any given line
PART = re.compile("part name='[0-9]+'")

BOX = re.compile("box top=")

# load the contents of the original XML file and open the output file
# for writing
print("[INFO] parsing data split XML file...")
rows = open(args["input"]).read().strip().split("\n")
output = open(args["output"], "w")

# loop over the rows of the data split file
for row in rows:
	# check to see if the current line has the (x, y)-coordinates for
	# the facial landmarks we are interested in
	parts = re.findall(PART, row)
	box = re.findall(BOX, row)

	if len(box) != 0:
		row = re.sub("'[0-9]+'", "'0'", row)
		output.write("{}\n".format(row))

	# elif len(parts) != 0:
	if len(parts) != 0:
		# parse out the name of the attribute from the row
		attr = "name='"
		i = row.find(attr)
		j = row.find("'", i + len(attr) + 1)
		name = int(row[i + len(attr):j])

		# if the facial landmark name exists within the range of our
		# indexes, write it to our output file
		if name in LANDMARKS:
			output.write("{}\n".format(row))
	else:
		output.write("{}\n".format(row))
# close the output file
output.close()