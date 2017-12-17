
# coding: utf-8

# In[35]:


import os
import sys
import collections
import itertools
import argparse
from shutil import copyfile


# In[36]:


parser = argparse.ArgumentParser()
parser.add_argument(
  '-i',
  '--input_prefix',
  dest='input',
  type=str,
  default='train',
  help='the prefix of input files')
parser.add_argument(
  '-o',
  '--output_prefix',
  dest='output',
  type=str,
  default='vocab',
  help='the prefix of output files')
parser.add_argument(
  '-v',
  '--vocab_size',
  dest='vocab_size',
  type=int,
  default=32000,
  help='limit the maximum vocab size to ...')
parser.add_argument(
  '-e',
  '--extention',
  dest='exts',
  type=str,
  default='jp,cn',
  help='comma segged extentions')
args = parser.parse_args('')


# In[40]:


try:
  os.mkdir('noshare')
except:
  print('noshare dir exist.')
for ext in args.exts.split(','):
  input_fpath=args.input+'.'+ext
  print(input_fpath)
  output_fpath='noshare/'+args.output+'.'+ext
  print(output_fpath)
  vocab=collections.Counter(w for l in open(input_fpath,'rt').readlines() for w in l.split())
  with open(output_fpath,'wt') as f:
    _=f.write('<unk>\n<s>\n</s>\n')
    for idx,w in enumerate(vocab.most_common(args.vocab_size)):
      _=f.write(w[0]+'\n')
    print('gen {} vocabs at {}'.format(idx+1,output_fpath))



# In[39]:


try:
  os.mkdir('share')
except:
  print('share dir exist.')

vocab = set(
  w
  for w in itertools.chain.from_iterable(
    open('noshare/' + args.output + '.' + ext, 'rt').readlines()
    for ext in args.exts.split(',')))
print(len(vocab))
with open('share/' + args.output + '.' + args.exts.split(',')[0], 'wt') as f:
  _ = f.write('<unk>\n<s>\n</s>\n')
  for idx, w in enumerate(vocab):
    if w not in ['<s>\n','<unk>\n','</s>\n']:
        _ = f.write(w)
  print('gen {} vocabs at {}'.format(
    idx + 1, 'share/' + args.output + '.' + args.exts.split(',')[0]))
copyfile('share/' + args.output + '.' + args.exts.split(',')[0],
         'share/' + args.output + '.' + args.exts.split(',')[1])
print('share/' + args.output + '.' + args.exts.split(',')[1])
print('finished')

