import os
import io
import glob
import json
import re
import pickle
from functools import reduce
from collections import defaultdict, Counter
from multiprocessing import Pool
from difflib import SequenceMatcher

import fire
import spacy
import numpy as np
from bs4 import BeautifulSoup
from spacy_langdetect import LanguageDetector
from termcolor import colored
from google.protobuf.json_format import MessageToJson
from PIL import Image


entity_black_list = [
    'Stock photography',
    'Image',
    'Getty Images',
    'Royalty-free',
    'stock.xchng',
    'Photography',
    'ストックフォト',
    'Photo caption',
    'Clip art',
    'Illustration',
    'Quotation mark',
    'Portrait',
    'ʻOkina',
    'Apostrophe',
    'Quotation',
    'Punctuation',
    'Black and white',
    'Quotation marks in English',
    'Text',
    'Internet meme',
    'Stock illustration',
    'photo library',
    'Portable Network Graphics',
    'Photograph',
    'iStock',
    'Graphics',
    'photo caption',
    'Hawaiian language'
    'alamy',
    'illustration',
    'Shutterstock',
    'Poster',
    'Facebook',
    'Royalty payment',
    'E-book',
    'jpeg',
    'png',
    'Logo',
    'Vector graphics',
    'cartoon',
    'YouTube',
]
entity_black_list = [e.lower() for e in entity_black_list]

entity_white_list = [ 
    'disable',
    'disability',
    'down syndrome',
    'downsyndrome',
    'immigran',
    'handicapped',
]

noun_chunk_blist = [
    'stock pictures',
    'stock picture',
    'stock photos',
    'stock photo',
    'premium high res pictures - getty image',
    'premium high res picture',
    'premium high res pictures - getty image',
    'photo and premium high res pictures - getty',
    'stock photo - download image',
    'high - res stock photo - getty image',
    '... - istock',
    '-pron-',
    'pictures',
    'photos',
    'premium high re',
    'royalty - free image',
    'royalty - free images - istock',
    'photo',
    'picture',
    'royalty - free photo & images - getty',
    'premium high',
    'royalty',
    'high resolution stock photography',
    'images - getty image',
    'royalty - free stock photo',
    'coronavirus',
    '- istock',
    '- photopin',
    'free image',
    'pinterest',
    'stock pictures',
    'pictures | getty image',
    'getty images',
    '- alamy',
    'royalty - free vector graphics',
    '- pinterest',
    'portrait photos',
    '- page',
    '- getty image',
    '- getty',
]
noun_chunk_blist = sorted(noun_chunk_blist, key=lambda x: len(x), reverse=True)


def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):
    """Return best matching substring of corpus.

    Parameters
    ----------
    query : str
    corpus : str
    step : int
        Step size of first match-value scan through corpus. Can be thought of
        as a sort of "scan resolution". Should not exceed length of query.
    flex : int
        Max. left/right substring position adjustment value. Should not
        exceed length of query / 2.

    Outputs
    -------
    output0 : str
        Best matching substring.
    output1 : float
        Match ratio of best matching substring. 1 is perfect match.
    """

    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m : m-1+qlen]))
            if verbose:
                print(query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)

    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted 
        # to optimize bmv_*
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        bmv_l = match_values[p_l // step]
        bmv_r = match_values[p_l // step]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print("\n" + str(f))
                print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))

        return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    if flex >= qlen/2:
        print("Warning: flex exceeds length of query / 2. Setting to default.")
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step

    pos_left, pos_right, match_value = adjust_left_right_positions()

    return slice(pos_left, pos_right), match_value



def detect_web(path):
    from google.cloud import vision
    """Detects web annotations given an image."""
    
    assert os.path.exists(path) and os.path.isfile(path)
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print('\nBest guess label: {}'.format(label.label))

    if annotations.pages_with_matching_images:
        print('\n{} Pages with matching images found:'.format(
            len(annotations.pages_with_matching_images)))

        # for page in annotations.pages_with_matching_images:
        #     print('\n\tPage url   : {}'.format(page.url))

        #     if page.full_matching_images:
        #         print('\t{} Full Matches found: '.format(
        #                len(page.full_matching_images)))

        #         for image in page.full_matching_images:
        #             print('\t\tImage url  : {}'.format(image.url))

        #     if page.partial_matching_images:
        #         print('\t{} Partial Matches found: '.format(
        #                len(page.partial_matching_images)))

        #         for image in page.partial_matching_images:
        #             print('\t\tImage url  : {}'.format(image.url))

    # if annotations.web_entities:
    #     print('\n{} Web entities found: '.format(
    #         len(annotations.web_entities)))

    #     for entity in annotations.web_entities:
    #         print('\n\tScore      : {}'.format(entity.score))
    #         print(u'\tDescription: {}'.format(entity.description))

    # if annotations.visually_similar_images:
    #     print('\n{} visually similar images found:\n'.format(
    #         len(annotations.visually_similar_images)))

    #     for image in annotations.visually_similar_images:
    #         print('\tImage url    : {}'.format(image.url))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    return annotations


def detect_image(path, json_path):
    annotations = detect_web(path)
    json_str = MessageToJson(annotations)
    with open(json_path, mode='w') as f:
        f.write(json_str)


def detect_dataset(img_list, output_dir, auto_break=20):
    os.makedirs(output_dir, exist_ok=True)
    start_id = -1
    with open(img_list, mode='r') as f:
        for i, line in enumerate(f):
            line = line.replace('\n', '')
            assert os.path.exists(line)
            print(f"[{i}] {line}")
            
            img_name = os.path.basename(line)
            json_name = img_name.replace('.jpg', '').replace('.png', '') + '.json'
            json_path = os.path.join(output_dir, json_name)

            if os.path.exists(json_path):
                print(f'Skip {line}, it already exists!')
            else:
                if start_id < 0:
                    start_id = i
                else:
                    if i - start_id >= auto_break:
                        break
                detect_image(line, json_path)
                print('-' * 100)


def create_img_list(img_dir, output_dir, split_size=30000, exclude_dir=None):
    if exclude_dir is not None:
        eimg_list = glob.glob(os.path.join(exclude_dir, '*.png'))
        eimg_list += glob.glob(os.path.join(exclude_dir, '**', '*.png'))
        eimg_list = [os.path.basename(ei).split('.')[0] for ei in eimg_list]
    else:
        eimg_list = []

    img_list = glob.glob(os.path.join(img_dir, '*.png'))
    img_list += glob.glob(os.path.join(img_dir, '**', '*.png'))
    print(f"Find {len(img_list)} images")
    img_list = [
        im for im in img_list
        if os.path.basename(im).split('.')[0] not in eimg_list
    ]
    print(f"Find {len(img_list)} images after filter")
    
    os.makedirs(output_dir, exist_ok=True)
    dir_name = os.path.basename(img_dir)
    
    for j, i in enumerate(range(0, len(img_list), split_size)):
        split = img_list[i: i+split_size]
        file_name = os.path.join(output_dir, f"{dir_name}_split.{j}.txt")
        with open(file_name, mode='w') as f:
            # f.writelines(split)
            for l in split:
                f.write(l + '\n')


def create_description(json_dir='/home/ron/Downloads/hateful_meme_data/web_entity_clean_all/', out_pickle=None):
    assert out_pickle is not None
    # json_dir = '/home/ron/Downloads/hateful_meme_data/web_entity/'
    json_list = glob.glob(os.path.join(json_dir, '*.json'))
    print(len(json_list))
    assert len(json_list) > 0, f"No json file founded in: {json_dir}"

    json_map = defaultdict(lambda: {'main': None, 'split': {}})
    name_pat = re.compile(r'(\d+)\.?(\d)?\.json')

    for i, j in enumerate(json_list):
        file_name = os.path.basename(j)
        m = re.match(name_pat, file_name)
        assert m is not None, j
        try:
            with open(j, 'r') as f:
                content = json.load(f)
                if m.group(2) is None:
                    json_map[m.group(1)]['main'] = content
                else:
                    json_map[m.group(1)]['split'][int(m.group(2))] = content
        except Exception as e:
            print(i, j, e)
            continue
    
    search_math = [0, 0]
    entity_map = defaultdict(lambda: {})
    title_map = defaultdict(lambda: {})
    count_entity = Counter()
    num_entity = []

    for k, d in json_map.items():
        img_serach = {0: d['main']} if len(d['split']) == 0 else d['split']
        
        for split_n, search in img_serach.items():
            if 'pagesWithMatchingImages' in search:
                search_math[0] += 1
        #         print(main['webEntities'])
                entity_name = [e['description'] for e in search['webEntities'] if 'description' in e]
                if 'label' in search['bestGuessLabels'][0]:
                    entity_name += [search['bestGuessLabels'][0]['label']]
                entity_name = [e for e in entity_name if e.lower() not in entity_black_list]
                titles = [
                    BeautifulSoup(page['pageTitle']).text
                    for page in search['pagesWithMatchingImages']
                    if 'pageTitle' in page
                ]
                
                entity_map[k][split_n] = entity_name
                title_map[k][split_n] = titles
                
                ent_count = Counter(entity_name)
                count_entity.update(ent_count)
                num_entity.append(len(entity_name))
            else:
                search_math[1] += 1

    num_entity = Counter(num_entity)

    with open(out_pickle, mode='wb') as pf: 
        pickle.dump({
            'entity_map': dict(entity_map),
            'title_map': dict(title_map),
            'json_map': dict(json_map),
        }, pf)


def sent_cluster(roberta, embed_titles):
    embed = roberta.encode(embed_titles)

    for i, t in enumerate(embed_titles):
        print(f"[{i}]", t)

    norm = np.linalg.norm(embed, keepdims=True, axis=1)
    mtx_norm = np.matmul(norm, norm.T)
    mtx_cos = np.matmul(embed, embed.T) / mtx_norm
    
    cluster_mark = [-1] * len(embed_titles)
    cluster_cnt = 0
    for i, row in enumerate(mtx_cos):
        if cluster_mark[i] == -1:
            cluster_mark[i] = cluster_cnt
            cluster_cnt += 1
        for j in range(i, len(row)):
            if row[j] > 0.5 and cluster_mark[j] == -1:
                cluster_mark[j] = cluster_mark[i]
    print(cluster_mark)
    cus_size = Counter(filter(lambda x: x >= 0, cluster_mark))
    cluster_id, _ = cus_size.most_common(1)[0]
    print(cluster_id, _)
    
    gather_sent = [t for t, c in zip(embed_titles, cluster_mark) if c == cluster_id]
    return gather_sent


def link_noun_chunk(token, token_map, direction=None, depth=0, prev_token=None):
    if depth > 3:
        return []
    
    tk = token
    print(
        list(tk.children),
        colored(' --> ', color='blue'),
        tk,
        colored(' --> ', color='blue'),
        f"({tk.head}, {tk.head.pos_})",
        tk.dep_
    )
    token_link = [token]
    if direction is None:
        if tk.dep_ in ['compound', 'amod', 'poss', 'part']:
            if tk.head.pos_ in ['ADJ', 'NOUN', 'PROPN', 'PART']:
                token_link += link_noun_chunk(tk.head, token_map, direction='head', depth=depth+1, prev_token=tk)
        if tk.children:
            for c in tk.children:
                if c.dep_ in ['poss', 'probj', 'amod', 'compound']:
                    token_link += link_noun_chunk(c, token_map, direction='child', depth=depth+1, prev_token=tk)
                    
    else:
        if tk.dep_ != 'ROOT' and direction == 'head':
            if tk.pos_ == 'ADP' and tk.dep_ == 'prep':
                token_link += link_noun_chunk(tk.head, token_map, direction='head', depth=depth+1, prev_token=tk)
            elif tk.dep_ in ['compound', 'dep', 'amod', 'poss', 'part']:
                token_link += link_noun_chunk(tk.head, token_map, direction='head', depth=depth+1, prev_token=tk)
        
        if tk.children:
            for c in tk.children:
                if c.dep_ in ['poss', 'compound'] and c != prev_token:
                    token_link += link_noun_chunk(c, token_map, direction='child', depth=depth+1, prev_token=tk)
    return token_link


nlp = spacy.load("en_core_web_lg")
def extract_subject(titles):
    # nlp = spacy.load("en_core_web_lg")
    global nlp
    entity_cnt = defaultdict(lambda: 0)
    token_maps = []
    
    for title in titles:
        doc = nlp(title)
        entity2chunk = defaultdict(list)
        token_map = {}
        
        for token in doc:
            if len(token.text) == 1:
                continue
            if token.pos_ in ['NOUN', 'PROPN']:
                entity2chunk[token.text] += token.children
                entity_cnt[token.text] += 1
                token_map[token.text] = token
        print(dict(entity2chunk))
        token_maps.append(token_map)
    
    print(dict(entity_cnt))
    entity_cnt_list = sorted(list(entity_cnt.items()), key=lambda x: x[1], reverse=True)
    select_subject = [w for w, c in entity_cnt_list[:2]]
    
    result_noun_chunks = []
    subj_chunks = []
    
    for token_map in token_maps:
        
        for subj in select_subject:
            try:
                result = link_noun_chunk(token_map[subj], token_map)
                subj_chunks.append(result)
                print(colored("##", color='yellow'), result)
            except KeyError:
                pass
            print('-' * 100)
        
    if len(subj_chunks) > 1:
        cluster_mark = [-1] * len(subj_chunks)
        for i in range(0, len(subj_chunks)):
            if cluster_mark[i] == -1:
                cluster_mark[i] = max(cluster_mark) + 1
                for j in range(i + 1, len(subj_chunks)):
                    i_txt = {t.text for t in subj_chunks[i]}
                    j_txt = {t.text for t in subj_chunks[j]}
                    print(len(i_txt.union(j_txt)), len(j_txt) + len(i_txt))
                    if len(i_txt.union(j_txt)) < len(j_txt) + len(i_txt):
                        cluster_mark[j] = cluster_mark[i]
        unify_chunks = []
        for i in range(max(cluster_mark) + 1):
            cs = [sc for sc, j in zip(subj_chunks, cluster_mark) if j == i]
            # print('cs: ', cluster_mark)
            cs = reduce(lambda a, b: a + b, cs)
            unify_chunks.append(cs)
        result_noun_chunks += unify_chunks
    else:
        result_noun_chunks += subj_chunks
    
    """
    Gather reuslt
    """
    result_sent = []
    for title in titles:
        contain_white = any([w in title for w in entity_white_list])
        if contain_white:
            result_sent.append({title})
    for tokens in result_noun_chunks:
        token_by_id = {}
        for t in tokens:
            token_by_id[t.i] = t
        result_sent.append({v.text for k, v in sorted(token_by_id.items(), key=lambda x: x[0])})
    return select_subject, result_sent


def titles_cleanup(img_entity_pickle, out_pickle=None):
    with open(img_entity_pickle, 'rb') as pf:
        imgs_web_entity = pickle.load(pf)
    title_map = imgs_web_entity['title_map']
    
    snlp = spacy.load("en_core_web_lg")
    snlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

    all_titles = []
    all_title_idx = []
    all_split_idx = []
    for id, imgs_titles in title_map.items():
        for n, img_titles in imgs_titles.items():
            for t in img_titles:
                all_titles.append(t.lower())
                all_title_idx.append(id)
                all_split_idx.append(n)

    assert len(all_titles) == len(all_title_idx), f"{len(all_titles)} != {len(all_title_idx)}"
    
    pipe = snlp.pipe(all_titles)
    clean_title_map = defaultdict(lambda: defaultdict(list))
    all_clean_title = []
    drop_by_lanu = 0

    for i, doc in enumerate(pipe):
        if doc._.language['language'] != 'en':
            drop_by_lanu += 1
            continue
        
        id = all_title_idx[i]
        senten = ''
        for token in doc:
            if token.pos_ != 'NUM' and token.pos_ != 'X':
                senten += token.text.lower() + ' '
        for b in noun_chunk_blist:
            senten = re.sub(b, '', senten)
        for b in entity_black_list:
            senten = re.sub(b, '', senten)
        
        n_split = all_split_idx[i]
        # if len(clean_title_map[id]) < n_split + 1:
        #     clean_title_map[id] += [[] for _ in range(n_split + 1 - len(clean_title_map[id]))]
        clean_title_map[id][n_split].append(senten)
        all_clean_title.append(senten)
        
    imgs_web_entity['clean_title_map'] = dict(clean_title_map)
    if out_pickle is None:
        out_pickle = img_entity_pickle
    with open(out_pickle, mode='wb') as pf: 
        pickle.dump(imgs_web_entity, pf)


def remove_duplicate(titles):
    dup = [False for _ in range(len(titles))]
    for i, title in enumerate(titles[:-1]):
        if dup[i]:
            continue
        for j, title_b in enumerate(titles[i + 1:]):
            # print(title, title_b)
            if len(title) > len(title_b):
                match_slice, match_score = get_best_match(title_b, title, step=2, flex=4)
            else:
                match_slice, match_score = get_best_match(title, title_b, step=2, flex=4)
            if match_score > 0.85:
                dup[j + i + 1] = True
    filtered = [t for d, t in zip(dup, titles) if not d]
    return filtered

def _apply_summary(args):
    id, n_split, titles = args
    print(colored(id, color='green'))
    titles = [re.sub('\|', '', e) for e in titles]
    titles = remove_duplicate(titles)
    select_subject, result_noun_chunks = extract_subject(titles)
    # title_summaries[id] = result_noun_chunks
    print('result_sent: ', result_noun_chunks)
    return id, n_split, result_noun_chunks

def titles_summary(img_entity_pickle, out_pickle):
    with open(img_entity_pickle, 'rb') as pf:
        imgs_web_entity = pickle.load(pf)
    clean_title_map = imgs_web_entity['clean_title_map']

    title_summaries = defaultdict(dict)
    with Pool(16) as pool:
        flatten = [
            (id, n_split, titles)
            for id, split_titles in clean_title_map.items()
            for n_split, titles in split_titles.items()
        ]
        results = pool.map(_apply_summary, flatten)
        for id, n_split, summary in results:
            # if len(title_summaries[id]) < n_split + 1:
            #     title_summaries[id] += [None for _ in range(n_split + 1 - len(title_summaries[id]))]
            title_summaries[id][n_split] = summary
    imgs_web_entity['title_summaries'] = title_summaries

    with open(out_pickle, mode='wb') as pf: 
        pickle.dump(imgs_web_entity, pf)


def insert_anno_jsonl(img_entity_pickle, anno_json, split_boxes_json, ocr_boxes_json, img_dir='/meme_data/img'):

    def refine_split_box(boxes, img_name):
        img_path = os.path.join(img_dir, img_name)
        w, h = Image.open(img_path).size
        
        if len(boxes) > 1:
            direction = None

            if w > h:
                direction = 'left-right'
                boxes = sorted(boxes, key=lambda x: x[0])
                for j in range(len(boxes) - 1):
                    boxes[j + 1][0] = boxes[j][2] = (boxes[j][2] + boxes[j + 1][0]) / 2
                
                boxes[0][0] = 0
                boxes[-1][2] = w
                for j in range(len((boxes))):
                    boxes[j][1] = 0
                    boxes[j][3] = h
            else:
                direction = 'top-down'
                boxes = sorted(boxes, key=lambda x: x[1])
                for j in range(len(boxes) - 1):
                    boxes[j][3] = boxes[j + 1][1] = (boxes[j][3] + boxes[j + 1][1]) / 2
            
                boxes[0][1] = 0
                boxes[-1][3] = h
                for j in range(len((boxes))):
                    boxes[j][0] = 0
                    boxes[j][2] = w
            return boxes
        else:
            return [[0, 0, w, h, 1.0]]
            
    
    def box_coverage(img_box, ocr_box):
        w = ocr_box[2] - ocr_box[0]
        h = ocr_box[3] - ocr_box[1]
        ocr_l_to_img_r = max(min(img_box[2] - ocr_box[0], w), 0)
        ocr_r_to_img_l = max(min(ocr_box[2] - img_box[0], w), 0)
        cover_w = min(ocr_l_to_img_r, ocr_r_to_img_l)
        
        ocr_t_to_img_b = max(min(img_box[3] - ocr_box[1], h), 0)
        ocr_b_to_img_t = max(min(ocr_box[3] - img_box[1], h), 0)
        cover_h = min(ocr_t_to_img_b, ocr_b_to_img_t)
        return (cover_h * cover_w) / (w * h)
        
    def merge_ocr_boxes(ocr_boxes):
        raise NotImplementedError

        box_id_pair = []
        font_height = []
        for i, box_info in enumerate(ocr_boxes):
            box, txt, score = box_info
            box_id_pair.append((i, box))
            font_height.append(box[3] - box[1])
        font_size = sum(font_height) / len(font_height)
        
        box_by_x = sorted(ocr_boxes, lambda box_info: box_info[0][0])
        box_by_xy = sorted(box_by_x, lambda box_info: box_info[0][1])
        box_by_y = sorted(ocr_boxes, lambda box_info: box_info[0][1])
        box_by_yx = sorted(box_by_y, lambda box_info: box_info[0][0])
        ln = len(box_by_xy)

        for i in range(ln - 1):
            box_by_xy[i]
            box_by_xy[i + 1]
    
    def find_appearence(meme_txt, ocr_boxes, img_boxes):
        box_by_x = sorted(ocr_boxes, key=lambda box_info: box_info[0][0])
        box_by_xy = sorted(box_by_x, key=lambda box_info: box_info[0][1])
        ocr_to_img_box = []
        
        for i, box_info in enumerate(box_by_xy):
            box, txt, score = box_info
            imbox_cover = [box_coverage(im, box) for im in img_boxes]
            argmax = imbox_cover.index(max(imbox_cover))
            ocr_to_img_box.append(argmax)
        print('ocr_to_img_box: ', ocr_to_img_box)
        
        for i, box_info in enumerate(box_by_xy):
            box, txt, score = box_info
            match_slice, match_score = get_best_match(txt, meme_txt, step=2, flex=4)
            print(
                colored(txt, color='blue'),
                colored(meme_txt[match_slice], color='green'),
                match_score,
                match_slice,
                len(meme_txt)
            )
            box_by_xy[i] = (box, txt, score, match_slice, match_score)

            if i > 0 and match_score >= 0.75:
                pass
                # assert match_slice.start >= box_by_xy[i - 1][3].stop
            elif match_score < 0.75:
                logger.warning(f"Low match score! {match_score}")
        char_to_img_box = [-1] * len(meme_txt)

        # last_im_box_id = -1
        for i, box_info in enumerate(box_by_xy):
            match_slice, match_score = box_info[3:5]
            match_len = match_slice.stop - match_slice.start
            if match_len < 6 and match_score < 0.75:
                continue
            elif match_score < 0.6:
                continue
            
            for j in range(match_slice.start, min(match_slice.stop, len(meme_txt))):
                char_to_img_box[j] = ocr_to_img_box[i]
            
            if i < len(box_by_xy) - 1:
                next_slice = box_by_xy[i - 1][3]
                for j in range(match_slice.stop, next_slice.start):
                    char_to_img_box[j] = ocr_to_img_box[i]
        print(char_to_img_box)
        for i in range(len(char_to_img_box)):
            c = char_to_img_box[i]
            if c < 0:
                next_match = len(char_to_img_box)
                next_img_id = max(char_to_img_box)
                for j in range(i, len(char_to_img_box)):
                    if char_to_img_box[j] >= 0:
                        next_match = j
                        next_img_id = char_to_img_box[j]
                        break
                prev_img_id = max(char_to_img_box[:i]) if i > 0 else -1

                for j in range(i, next_match):
                    if prev_img_id == next_img_id:
                        char_to_img_box[j] = prev_img_id
                    elif prev_img_id < next_img_id:
                        char_to_img_box[j] = prev_img_id + 1
                    else:
                        # NOTE: happend when ocr's detected text not matching actual anno text.
                        if j > (i + next_match) // 2:
                            char_to_img_box[j] = next_img_id
                        else:
                            char_to_img_box[j] = prev_img_id
                    #     raise RuntimeError(f'!? {prev_img_id}, {next_img_id}')
                
        
        print(char_to_img_box)
        print('-' * 100)
        return char_to_img_box
    
    meme_anno = []
    
    with open(anno_json, 'r') as f:
        for line in f:
            meme_anno.append(json.loads(line))
    
    with open(img_entity_pickle, 'rb') as pf:
        imgs_web_entity = pickle.load(pf)
    
    with open(split_boxes_json, 'r') as f:
        image_split_annos = json.load(f)
    
    with open(ocr_boxes_json, 'r') as f:
        ocr_anno = json.load(f)
    
    title_summaries = imgs_web_entity['title_summaries']
    entity_map = imgs_web_entity['entity_map']

    for anno in meme_anno:
        id = anno['id']
        img_name = os.path.basename(anno['img'])

        image_boxes = image_split_annos[img_name]
        image_boxes = refine_split_box(image_boxes, img_name)
        ocr_boxes = ocr_anno[img_name]
        
        if len(image_boxes) > 1:
            logger.info(img_name)
            char_to_img_box = find_appearence(anno['text'], ocr_boxes, image_boxes)
        else:
            char_to_img_box = [0] * len(anno['text'])

        if f"{id:05d}" in entity_map:
            img_entitys_splits = entity_map[f"{id:05d}"]
            # img_entitys = reduce(lambda a, b: a + b, img_entitys_splits)
            summaries_splits = title_summaries[f"{id:05d}"]
        else:
            if f"{id:05d}" in title_summaries:
                summaries_splits = title_summaries[f"{id:05d}"]
                img_entitys_splits = {k: [] for k in summaries_splits.keys()}
            else:
                summaries_splits = {}
                img_entitys_splits = {}

        anno['image_partition'] = []
        anno['partition_description'] = []
        anno['text_char_partition_id'] = char_to_img_box
        assert max(char_to_img_box) < len(image_boxes), f"{max(char_to_img_box)} !< {len(image_boxes)}"

        # split_number = sorted(img_entitys_splits.keys())
        # for sn in split_number:
        #     anno['image_partition'].append(image_boxes[sn])
        #     entitys = img_entitys_splits[sn]
        #     if len(entitys) <= 2:
        #         if sn in summaries_splits:
        #             web_page_summaries = summaries_splits[sn]
        #             entitys += [' '.join(s) for s in web_page_summaries]
        #     anno['partition_description'].append(entitys)
        # assert len(anno['image_partition']) == len(image_boxes), F"{img_name}"
        
        anno['image_partition'] = image_boxes
        for sn in range(len(image_boxes)):
            if sn in img_entitys_splits:
                entitys = img_entitys_splits[sn]
            else:
                entitys = []
            use_title = len(entitys) <= 2
            if use_title:
                if sn in summaries_splits:
                    web_page_summaries = summaries_splits[sn]
                    entitys += [' '.join(s) for s in web_page_summaries]
            anno['partition_description'].append(entitys)
    
    out_path = anno_json.replace('.json', '.entity.json')
    with open(out_path, 'w') as f:
        for anno_line in meme_anno:
            seri_line = json.dumps(anno_line)
            f.write(f"{seri_line}\n")


if __name__ == "__main__":
    from loguru import logger
    """
    create_img_list --> detect_dataset --> create_description --> titles_cleanup 
    --> titles_summary  --> insert_anno_jsonl
    """
    
    with logger.catch(reraise=True):
        fire.Fire({
            'detect_web': detect_web,
            'detect_image': detect_image,
            'detect_dataset': detect_dataset,
            'create_img_list': create_img_list,
            'create_description': create_description,
            'titles_cleanup': titles_cleanup,
            'titles_summary': titles_summary,
            'insert_anno_jsonl': insert_anno_jsonl,
        })