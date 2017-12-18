import os,sys
import pickle

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

comp_dict=pickle.load(open(os.path.join(CURRENT_DIR,'dicts','comp_dict.pkl'),"rb"))
stroke_dict = pickle.load(
    open(os.path.join(CURRENT_DIR, 'dicts', 'stroke_dict.pkl'), "rb"))
