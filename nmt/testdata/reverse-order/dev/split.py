import os

CURRENT_DIR = os.path.dirname(__file__)
with open(os.path.join(CURRENT_DIR, 'data.txt'), 'rt') as f, open(
        os.path.join(CURRENT_DIR, 'data.source'), 'wt') as f_source, open(
            os.path.join(CURRENT_DIR, 'data.target'), 'wt') as f_target:
    for line in f:
        s, t = line.strip().split('\t')
        f_source.write(s + '\n')
        f_target.write(t + '\n')
