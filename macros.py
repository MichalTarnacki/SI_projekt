import re
train_path = "media/train/"
test_path = "media/test/"
model_path = "media/models/"
spectros_path = "media/spectros/"
train_breaths = "media/sorted/in/"
train_exhales = "media/sorted/out/"
sorted_path = "media/sorted/"
test_sorted_path = "media/sorted_test/"
test_breaths = "media/sorted_test/in/"
test_exhales = "media/sorted_test/out/"
background_path = "media/background/"
background_sorted_path = "media/sorted/background/"
test_background_path = "media/test_bg/"
test_background_sorted_path = "media/sorted_test/background/"
freg = re.compile(r'^e[0-9]+$')