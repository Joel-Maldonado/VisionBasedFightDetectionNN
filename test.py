import os
import shutil
import numpy as np

DIR = '/u/jruiz_intern/jruiz/Datasets/CombinedViolence'

categories = os.listdir(DIR)

v_vids = []
nv_vids = []

for category in categories:
  path = os.path.join(DIR, category)
  for video in os.listdir(path):
    if categories.index(category) == 0:
      nv_vids.append(os.path.join(path, video))
    else:
      v_vids.append(os.path.join(path, video))

np.random.shuffle(v_vids)
np.random.shuffle(nv_vids)

print(len(v_vids))
print(len(nv_vids))

TRAIN_SPLIT = int((len(v_vids) + len(nv_vids)) * 0.8)//2

for i in range(0, TRAIN_SPLIT):
  shutil.copy(v_vids[i], 'CombinedViolence/train/1_violent')
  shutil.copy(nv_vids[i], 'CombinedViolence/train/0_non_violent')
  print(f'Copied {i*2}/{TRAIN_SPLIT*2}')

for i in range(TRAIN_SPLIT, len(v_vids)):
  shutil.copy(v_vids[i], 'CombinedViolence/test/1_violent')
  shutil.copy(nv_vids[i], 'CombinedViolence/test/0_non_violent')
  print(f'Copied {i*2}/{len(v_vids*2)}')

# print(len(videos))

# TRAIN_SPLIT = int(len(videos) * 0.8)
# print(TRAIN_SPLIT)


# np.random.shuffle(videos)

# TRAIN_VIDEOS = videos[:TRAIN_SPLIT]
# TEST_VIDEOS = videos[TRAIN_SPLIT:]

# # print(len(TRAIN_VIDEOS))
# # print(len(TEST_VIDEOS))
# # print(TRAIN_VIDEOS[0])
# # print(TEST_VIDEOS[0])

# for i, video in enumerate(TRAIN_VIDEOS):
#   shutil.copy2(video[0], os.path.join('CombinedViolence/train', video[1]))
#   print(f'Train: {i}/{len(TRAIN_VIDEOS)}')

# for i, video in enumerate(TEST_VIDEOS):
#   shutil.copy2(video[0], os.path.join('CombinedViolence/test', video[1]))
#   print(f'Test: {i}/{len(TEST_VIDEOS)}')