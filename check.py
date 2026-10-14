import json
from pathlib import Path
from collections import Counter

snap_dir = Path('graph_model/snapshots')
files = sorted(snap_dir.glob('window_*.json'))
labels = []
sources = []
for f in files:
    with open(f) as fh:
        s = json.load(fh)
    labels.append(s['label'])
    sources.append(s['label_source'])

print('Total windows:', len(files))
print('Labels:', Counter(labels))
print('Sources:', Counter(sources))