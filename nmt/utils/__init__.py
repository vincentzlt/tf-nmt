import os,pickle,Mykytea

current_dir = os.path.dirname(__file__)
dict_file = os.path.join(current_dir, './dicts/comp_dict.pkl')
comp_dict = pickle.load(open(dict_file, 'rb'))

dict_file = os.path.join(current_dir, './dicts/stroke_dict.pkl')
stroke_dict = pickle.load(open(dict_file, 'rb'))
sentence = sentence.strip()

opt_jp = "-model /home/vincentzlt/kytea/models/jp-0.4.7-1.mod"

opt_cn = "-model /home/vincentzlt/kytea/models/msr-0.4.0-1.mod"

mk_cn = lambda x: list(Mykytea.Mykytea(opt_cn).getWS(x))

mk_jp = lambda x: list(Mykytea.Mykytea(opt_jp).getWS(x))

mk_else = lambda x: x.split()
