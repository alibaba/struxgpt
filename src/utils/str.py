# Copyright (c) Alibaba, Inc. and its affiliates.
import re
import string
import numpy as np
from collections import Counter
import jieba
from tqdm import tqdm
from typing import List

PUNCTUATION = string.punctuation + r'。，？：；！、【】｜'


def is_zh(s: str, ratio=0.3):
    # return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4DBF]', s))
    return len(re.findall(r'[\u4e00-\u9fff\u3400-\u4DBF]', s)) / len(s) > ratio


def is_empty(s: str):
    return len(white_space_fix(s)) == 0


def remove_punc(text: str, lowercase=False):
    if not text:
        return text
    text = text[:-1] if text[-1] in PUNCTUATION and text[-1] not in '>】’”)}]\'"' else text
    if lowercase:
        word0 = text.split()[0]
        if word0[1:].lower() != word0[1:]:
            pass
        else:
            text = text[:1].lower() + text[1:]
    return text


def set_punc(text: str, end='.'):
    if not text:
        return text
    return (text + end) if text[-1] not in PUNCTUATION else text


def upper_start(text: str, flag=True):
    l = len(text)
    if l == 0:
        return text
    if flag:
        return text[:1].upper() + text[1:]
    elif l == 1 or text[:2].upper() != text[:2]:
        return text[:1].lower() + text[1:]
    else:
        return text


def drop_last_segment(text: str, sep: str='_', num: int=1, last_mapping=lambda x: x):
    segments = text.split(sep)
    return sep.join(segments[:-num]), last_mapping(sep.join(segments[-num:]))


def white_space_fix(text: str):
    return " ".join(text.split())


def split_to_sentence(s: str):    
    # s = "Mr. Smith bought cheapsite.com for 1.5 million dollars, i.e. he paid a lot for it. Did he mind? Adam Jones Jr. thinks he didn't. In any case, this isn't true... Well, with a probability of .9 it isn't."
    # s = 'The Saronic Gulf (Greek:  , Saroniks klpos) or Gulf of Aegina in Greece is formed between the peninsulas of Attica and Argolis and forms part of the Aegean Sea. It defines the eastern side of the isthmus of Corinth, being the eastern terminus of the Corinth Canal, which cuts across the isthmus. The Saronic Islands in the gulf have played a pivotal role in the history of Greece, with the largest, Salamis, naming a significant naval battle in the Greco-Persian wars. The Megara Gulf makes up the northern end of the Saronic Gulf.'
    sentences = re.split('(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=\.|\?|\!)\"*\s*\s*(?:\W*)([A-Z])', s)
    sentences_new: List[str] = []
    for sentence in sentences:
        if len(sentences_new) and len(sentences_new[-1]) == 1 and sentences_new[-1].isupper():
            sentences_new[-1] += sentence
        else:
            sentences_new.append(sentence)
    sentences = sentences_new

    sentences = [white_space_fix(line) for line in sentences]
    sentences = [sentence for sentence in sentences if len(sentence)]

    return sentences


def split_to_sentence_zh(s: str):
    sent_ending = r'。：；\;！\!？\?'
    half_ending = r'”’）\)】\]》\>'
    s = re.sub('([%s])([^%s])' % (sent_ending, half_ending), r"\1\n\2", s)  # 单字符断句符
    s = re.sub('(\.{6})([^%s])' % half_ending, r"\1\n\2", s)  # 英文省略号
    s = re.sub('(\…{2})([^%s])' % half_ending, r"\1\n\2", s)  # 中文省略号
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    s = re.sub('([%s][%s])([^，%s])' % (sent_ending, half_ending, sent_ending), r'\1\n\2', s)
    s = s.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。

    return s.split("\n")


def normalize_statement(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, term='f1score'):
    normalized_prediction = normalize_statement(prediction)
    normalized_ground_truth = normalize_statement(ground_truth)

    prediction = normalized_prediction.split()
    ground_truth = normalized_ground_truth.split()

    res_dict = {'f1score': 0, 'recall': 0, 'precision': 0, 'all': [0, 0, 0]}

    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same != 0:
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)

        res_dict['precision'], res_dict['recall'], res_dict['f1score'] = precision, recall, f1
        res_dict['all'] = [precision, recall, f1]
    
    return res_dict[term]


def batched_f1_score(prediction_list, ground_truth_list, term='f1score', show_progress=True, prediction_counter_list=None):
    def generate_1gram(text):
        return Counter(normalize_statement(text).split())

    if prediction_counter_list is None:
        if show_progress:
            prediction_list = tqdm(prediction_list, desc='processing prediction')
        prediction_counter_list = [generate_1gram(prediction) for prediction in prediction_list]

    if show_progress:
        ground_truth_list = tqdm(ground_truth_list, desc='processing ground-truth')
    ground_truth_counter_list = [generate_1gram(ground_truth) for ground_truth in ground_truth_list]

    n, m = len(prediction_counter_list), len(ground_truth_counter_list)
    res_dict = {k: np.zeros((n, m)) for k in ['precision', 'recall', 'f1score']}
    pbar = tqdm(prediction_counter_list, desc='rouge cumputing') if show_progress else prediction_counter_list 
    for i, prediction_counter in enumerate(pbar):
        for j, ground_truth_counter in enumerate(ground_truth_counter_list):
            common = prediction_counter & ground_truth_counter
            num_same = sum(common.values())

            if num_same != 0:
                precision = 1.0 * num_same / sum(prediction_counter.values())
                recall = 1.0 * num_same / sum(ground_truth_counter.values())
                f1 = (2 * precision * recall) / (precision + recall)

                res_dict['precision'][i, j] = precision
                res_dict['recall'][i, j] = recall
                res_dict['f1score'][i, j] = f1
    
    return res_dict[term], prediction_counter_list


def batched_ngram_overlap(text_list1, text_list2, n, show_progress=True, ngram_list1=None):
    def generate_ngrams(text, n):
        return Counter(zip(*[text[i:] for i in range(n)]))
    
    if ngram_list1 is None:
        if show_progress:
            text_list1 = tqdm(text_list1, desc='processing text1')
        ngram_list1 = [generate_ngrams(text, n) for text in text_list1]

    if show_progress:
        text_list2 = tqdm(text_list2, desc='processing text2')
    ngram_list2 = [generate_ngrams(text, n) for text in text_list2]

    _n, _m = len(text_list1), len(text_list2)
    res_matrix = np.zeros((_n, _m))
    pbar = tqdm(ngram_list1, desc=f'{n}-gram computing') if show_progress else ngram_list1
    repreat_sents = []
    for i, ngram1 in enumerate(pbar):
        for j, ngram2 in enumerate(ngram_list2):
            overlap = ngram1 & ngram2
            overlap_ratio = sum(overlap.values()) / float(sum(ngram1.values()) + sum(ngram2.values()))
            res_matrix[i, j] = overlap_ratio

            if overlap_ratio > 0:
                repreat_sents.extend([k for k, v in overlap.items() if v > 0])

    return res_matrix, ngram_list1, repreat_sents


def batched_sentence_overlap(text_list1, text_list2, show_progress=True, text_line_counter1=None):
    from nltk.tokenize import sent_tokenize
    
    if text_line_counter1 is None:
        if show_progress:
            text_list1 = tqdm(text_list1, desc='processing text1')
        text_line_counter1 = [Counter(sent_tokenize(text)) for text in text_list1]
    
    if show_progress:
        text_list2 = tqdm(text_list2, desc='processing text2')
    text_line_counter2 = [Counter(sent_tokenize(text)) for text in text_list2]

    n, m = len(text_list1), len(text_list2)
    res_matrix = np.zeros((n, m))
    pbar = tqdm(text_line_counter1, desc='sentence computing') if show_progress else text_line_counter1
    repreat_sents = []
    for i, line_counter1 in enumerate(pbar):
        for j, line_counter2 in enumerate(text_line_counter2):
            overlap = line_counter1 & line_counter2
            union = line_counter1 | line_counter2
            num_same = sum(overlap.values())

            if num_same > 0:
                num_union = sum(union.values())
                assert num_union != 0
                res_matrix[i, j] = num_same / num_union

                repreat_sents.extend([k for k, v in overlap.items() if v > 0])
             
    return res_matrix, text_line_counter1, repreat_sents


def normalize_zh_statement(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_zh_score(prediction, ground_truth, term='f1score'):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_statement(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_statement(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    
    res_dict = {'f1score': 0, 'recall': 0, 'precision': 0, 'all': [0, 0, 0]}

    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same != 0:
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)

        res_dict['precision'], res_dict['recall'], res_dict['f1score'] = precision, recall, f1
        res_dict['all'] = [precision, recall, f1]
    
    return res_dict[term]


def normalize_statement_auto(s):
    func = normalize_zh_statement if is_zh(s) else normalize_statement
    return func(s)

def f1_score_auto(prediction, ground_truth, term='f1score'):
    func = f1_zh_score if is_zh(ground_truth) else f1_score
    return func(prediction, ground_truth, term=term)