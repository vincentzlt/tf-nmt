# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess

import tensorflow as tf

import Mykytea
import pickle

from ..scripts import bleu
from ..scripts import rouge

import pdb

__all__ = ["evaluate"]

if 'comp_dict' not in locals():
    current_dir = os.path.dirname(__file__)
    dict_file = os.path.join(current_dir, './dicts/comp_dict.pkl')
    comp_dict = pickle.load(open(dict_file, 'rb'))
if 'comp_dict' not in locals():
    current_dir = os.path.dirname(__file__)
    dict_file = os.path.join(current_dir, './dicts/stroke_dict.pkl')
    stroke_dict = pickle.load(open(dict_file, 'rb'))

opt_jp = "-model /home/vincentzlt/kytea/models/jp-0.4.7-1.mod"
opt_cn = "-model /home/vincentzlt/kytea/models/msr-0.4.0-1.mod"
if 'mk_cn' not in locals():
    mk_jp = lambda x: list(Mykytea.Mykytea(opt_cn).getWS(x))
if 'mk_jp' not in locals():
    mk_jp = lambda x: list(Mykytea.Mykytea(opt_jp).getWS(x))
if 'mk_else' not in locals():
    mk_else = lambda x: x.split()

def evaluate(ref_file,
             trans_file,
             metric,
             src,
             tgt,
             subword_option=None,
             text_format=None):
    """Pick a metric and evaluate depending on task."""
    # BLEU scores for translation task
    if metric.lower() == "bleu":
        evaluation_score = _bleu(
            ref_file,
            trans_file,
            subword_option=subword_option,
            text_format=text_format)
    # BLEU scores with char segmentation for translation task
    elif metric.lower() == "char_bleu":
        evaluation_score = _char_bleu(
            ref_file,
            trans_file,
            subword_option=subword_option,
            text_format=text_format)
    # BLEU scores with kytea segmentation for translation task
    elif metric.lower() == "kytea_bleu":
        evaluation_score = _kytea_bleu(
            ref_file,
            trans_file,
            src,
            tgt,
            subword_option=subword_option,
            text_format=text_format)
    # ROUGE scores for summarization tasks
    elif metric.lower() == "rouge":
        evaluation_score = _rouge(
            ref_file,
            trans_file,
            subword_option=subword_option,
            text_format=text_format)
    elif metric.lower() == "accuracy":
        evaluation_score = _accuracy(ref_file, trans_file)
    elif metric.lower() == "word_accuracy":
        evaluation_score = _word_accuracy(ref_file, trans_file)
    else:
        raise ValueError("Unknown metric %s" % metric)

    return evaluation_score


def _clean(sentence, subword_option, text_format):
    """Clean and handle BPE or SPM outputs."""
    sentence = sentence.strip()

    # BPE
    if subword_option == "bpe":
        sentence = re.sub("@@ ", "", sentence)
        if text_format == 'comp':
            sentence = u' '.join([comp_dict[w] for w in sentence.split()])
        elif text_format == 'stroke':
            sentence = u' '.join([stroke_dict[w] for w in sentence.split()])

    # SPM
    elif subword_option == "spm":
        sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()
        if text_format == 'comp':
            sentence = u' '.join([comp_dict[w] for w in sentence.split()])
        elif text_format == 'stroke':
            sentence = u' '.join([stroke_dict[w] for w in sentence.split()])



    return sentence


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file, subword_option=None,text_format=None):
    """Compute BLEU scores and handling BPE."""
    max_order = 4
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(
            tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option,text_format)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            line = _clean(line, subword_option=None,text_format=None)
            translations.append(line.split(" "))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


def _rouge(ref_file, summarization_file, subword_option=None,text_format=None):
    """Compute ROUGE scores and handling BPE."""

    references = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fh:
        for line in fh:
            references.append(_clean(line, subword_option,text_format))

    hypotheses = []
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(summarization_file, "rb")) as fh:
        for line in fh:
            hypotheses.append(_clean(line, subword_option=None,text_format=None))

    rouge_score_map = rouge.rouge(hypotheses, references)
    return 100 * rouge_score_map["rouge_l/f_score"]


def _accuracy(label_file, pred_file):
    """Compute accuracy, each line contains a label."""

    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
            count = 0.0
            match = 0.0
            for label in label_fh:
                label = label.strip()
                pred = pred_fh.readline().strip()
                if label == pred:
                    match += 1
                count += 1
    return 100 * match / count


def _word_accuracy(label_file, pred_file):
    """Compute accuracy on per word basis."""

    with codecs.getreader("utf-8")(
            tf.gfile.GFile(label_file, "r")) as label_fh:
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(pred_file, "r")) as pred_fh:
            total_acc, total_count = 0., 0.
            for sentence in label_fh:
                labels = sentence.strip().split(" ")
                preds = pred_fh.readline().strip().split(" ")
                match = 0.0
                for pos in range(min(len(labels), len(preds))):
                    label = labels[pos]
                    pred = preds[pos]
                    if label == pred:
                        match += 1
                total_acc += 100 * match / max(len(labels), len(preds))
                total_count += 1
    return total_acc / total_count


def _moses_bleu(multi_bleu_script, tgt_test, trans_file, subword_option=None):
    """Compute BLEU scores using Moses multi-bleu.perl script."""

    # TODO(thangluong): perform rewrite using python
    # BPE
    if subword_option == "bpe":
        debpe_tgt_test = tgt_test + ".debpe"
        if not os.path.exists(debpe_tgt_test):
            # TODO(thangluong): not use shell=True, can be a security hazard
            subprocess.call(
                "cp %s %s" % (tgt_test, debpe_tgt_test), shell=True)
            subprocess.call("sed s/@@ //g %s" % (debpe_tgt_test), shell=True)
        tgt_test = debpe_tgt_test
    elif subword_option == "spm":
        despm_tgt_test = tgt_test + ".despm"
        if not os.path.exists(despm_tgt_test):
            subprocess.call("cp %s %s" % (tgt_test, despm_tgt_test))
            subprocess.call("sed s/ //g %s" % (despm_tgt_test))
            subprocess.call(u"sed s/^\u2581/g %s" % (despm_tgt_test))
            subprocess.call(u"sed s/\u2581/ /g %s" % (despm_tgt_test))
        tgt_test = despm_tgt_test
    cmd = "%s %s < %s" % (multi_bleu_script, tgt_test, trans_file)

    # subprocess
    # TODO(thangluong): not use shell=True, can be a security hazard
    bleu_output = subprocess.check_output(cmd, shell=True)

    # extract BLEU score
    m = re.search("BLEU = (.+?),", bleu_output)
    bleu_score = float(m.group(1))

    return bleu_score


def _char_bleu(ref_file, trans_file, subword_option=None, text_format=None):
    """Compute BLEU scores and handling BPE."""
    max_order = 4
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option, text_format)
            reference_list.append(list(u''.join(reference.split(" "))))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            line = _clean(line, subword_option=None, text_format=None)
            translations.append(list(u''.join(line.split(" "))))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


def _char_bleu(ref_file, trans_file, subword_option=None, text_format=None):
    """Compute BLEU scores and handling BPE."""
    max_order = 4
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option, text_format)
            reference_list.append(list(u''.join(reference.split(" "))))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            line = _clean(line, subword_option=None, text_format=None)
            translations.append(list(u''.join(line.split(" "))))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score





def _kytea_bleu(ref_file,
                trans_file,
                src,
                tgt,
                subword_option=None,
                text_format=None):
    """Compute BLEU scores and handling BPE."""
    max_order = 4
    smooth = False


    if 'cn' in tgt:
      mk_tgt=mk_cn
    elif 'jp' in tgt:
      mk_tgt=mk_jp
    else:
      mk_tgt=mk_else


    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option, text_format)
            reference_list.append(mk_tgt(reference))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            line = _clean(line, subword_option=None, text_format=None)
            translations.append(mk_tgt(reference))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score
