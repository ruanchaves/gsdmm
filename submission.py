from gsdmm import MovieGroupProcess
import pandas as pd
import numpy as np
import spacy
from collections import Counter
import re
import pickle

from sklearn.metrics.cluster import contingency_matrix
import munkres
from sklearn.metrics import adjusted_rand_score

def train_mgp(spacy_lang='pt_core_news_sm', train_file='train.csv', category_column='category', title_column='title', number_regex='[0-9]', number_code='NUMBER', model_file='mgp.model', scores_file='scores.npy'):
    nlp = spacy.load(spacy_lang)
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    df = pd.read_csv(train_file)
    len_categories = len(df[category_column].drop_duplicates().values.tolist())
    mgp = MovieGroupProcess(K=len_categories+100, alpha=0.1, beta=0.1, n_iters=10)
    docs = df[title_column].values.tolist()
    tokens = []
    for item in docs:
        processed_item = re.sub(number_regex, number_code, item.lower())
        tmp = tokenizer(processed_item)
        tokens.append([str(x) for x in tmp if not (x.is_punct or x.is_stop)])

    tokens_freq_dict = dict(
        Counter([x for y in tokens for x in y]).most_common())
    for idx, item in enumerate(tokens):
        tokens[idx] = list(
            filter(lambda x: tokens_freq_dict[x] > 1, tokens[idx]))

    vocab_size = len(set(x for y in tokens for x in y))
    y = mgp.fit(tokens, vocab_size)
    scores = []
    for item in tokens:
        scores.append(np.array(mgp.score(item)))
    scores = np.array(scores)
    with open(model_file, 'wb') as f:
        pickle.dump(mgp, f)
        f.close()
    np.save(scores_file, scores)

def test_mgp(spacy_lang='pt_core_news_sm', test_file='test.csv', category_column='category', title_column='title', number_regex='[0-9]', number_code='NUMBER', model_file='mgp.model', scores_file='scores.npy'):
    nlp = spacy.load(spacy_lang)
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    df = pd.read_csv(test_file)
    docs = df[title_column].values.tolist()
    tokens = []
    for item in docs:
        processed_item = re.sub(number_regex, number_code, item.lower())
        tmp = tokenizer(processed_item)
        tokens.append([str(x) for x in tmp if not (x.is_punct or x.is_stop)])

    tokens_freq_dict = dict(
        Counter([x for y in tokens for x in y]).most_common())
    for idx, item in enumerate(tokens):
        tokens[idx] = list(
            filter(lambda x: tokens_freq_dict[x] > 1, tokens[idx]))

    with open(model_file, 'rb') as f:
        mgp = pickle.load(f)

    scores = []
    for item in tokens:
        scores.append(np.argmax(np.array(mgp.score(item))))
    scores = np.array(scores)
    return scores

if __name__ == '__main__':
    scores_file = 'scores.npy'
    model_file = 'mgp.model'
    train_file = 'train.csv'
    try:
        open(model_file, 'rb').close()
    except:
        train_mgp(scores_file=scores_file,
                  model_file=model_file, train_file=train_file)

    labels = pd.read_csv(train_file)['category'].values.tolist()
    label_dict = {v: i for i, v in enumerate(sorted(list(set(labels))))}
    rev_label_dict = {v: i for i, v in label_dict.items()}

    numeric_labels = [label_dict[x] for x in labels]

    scores = np.load(scores_file)
    max_scores = []
    for item in scores:
        max_scores.append(np.argmax(item))

    assert( len(numeric_labels) == len(max_scores) )

    labels_classes, labels_class_idx = np.unique(numeric_labels, return_inverse=True)
    max_scores_classes, max_scores_class_idx = np.unique(max_scores, return_inverse=True)
    cm = contingency_matrix(max_scores, numeric_labels)
    cm_argmax = np.argmax(cm, axis=1)
    translation_dict = {}
    for idx, item in enumerate(cm_argmax):
        translation_dict[max_scores_classes[idx]] = labels_classes[item]

    test_scores = test_mgp()

    preds = []
    for idx, item in enumerate(test_scores):
        num_code = translation_dict[item]
        preds.append({'id': idx, 'category': rev_label_dict[num_code]})

    pd.DataFrame(preds)[['id', 'category']].to_csv('submission.csv', index=False)
