import os
import torch
import numpy as np
import pandas as pd
import networkx as nx

from loguru import logger
from graph_tool.all import *
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaModel


def build_statement_nx_graph(data_dir, graph_out):
    assert os.path.exists(data_dir)
    item_tab = pd.read_csv(os.path.join(data_dir, 'item.csv'))
    statem = pd.read_csv(os.path.join(data_dir, 'statements.csv'))
    prop = pd.read_csv(os.path.join(data_dir, 'property.csv'))

    G = nx.Graph()
    num_items = len(item_tab)
    for i, row in item_tab.iterrows():
        G.add_node(row.item_id, label=row.en_label)
        print(f"[ITEM] {i}/{num_items}")
        if i > 100:
            break
    
    num_state = len(statem)
    for i, row in statem.iterrows():
        pid = row.edge_property_id
        plabel = prop.loc[prop['property_id'] == pid].en_label
        plabel = plabel.to_list()[0]
        print(f"[EDGE] {i}/{num_state}  {row.source_item_id} --[{pid}, {plabel}]--> {row.target_item_id}")
        G.add_edge(
            row.source_item_id,
            row.target_item_id,
            property_id=pid,
            property_label=plabel
        )
        if i > 100:
            break
    nx.write_gpickle(G, graph_out)


def build_statement_gtool_graph(data_dir, graph_out):
    assert os.path.exists(data_dir)
    save_dir = os.path.dirname(graph_out)
    log_file = os.path.join(save_dir, 'wiki_gtool_graph_{time}.log')
    logger.add(log_file)
    
    item_tab = pd.read_csv(os.path.join(data_dir, 'item.csv'))
    num_items = len(item_tab)

    G = Graph()
    v_label = G.new_vertex_property('string')
    v_id = G.new_vertex_property('int')
    id_to_vert = {}
    
    for i, row in item_tab.iterrows():
        v = G.add_vertex()
        v_id[v] = row.item_id
        v_label[v] = row.en_label
        id_to_vert[row.item_id] = v
        print(f"[ITEM] {i}/{num_items}")
        # if i > 100:
        #     break
    G.vertex_properties["label"] = v_label
    G.vertex_properties["id"] = v_id
    G.save(os.path.join(save_dir, 'wikidata.item.xml.gz'))
    del item_tab
    
    statem = pd.read_csv(os.path.join(data_dir, 'statements.csv'))
    prop = pd.read_csv(os.path.join(data_dir, 'property.csv'))
    num_state = len(statem)

    e_id = G.new_edge_property('int')
    e_label = G.new_edge_property('string')

    for i, row in statem.iterrows():
        pid = row.edge_property_id
        plabel = prop.loc[prop['property_id'] == pid].en_label
        plabel = plabel.to_list()[0]
        print(f"[EDGE] {i}/{num_state}  {row.source_item_id} --[{pid}, {plabel}]--> {row.target_item_id}")
        try:
            e = G.add_edge(
                id_to_vert[row.source_item_id],
                id_to_vert[row.target_item_id],
            )
            e_id[e] = pid
            e_label[e] = plabel
        except KeyError:
            logger.warning(f"{row.source_item_id} --[{pid}, {plabel}]--> {row.target_item_id} have missing item!")
        # if i > 100:
        #     break

    G.edge_properties["property_label"] = e_label
    G.edge_properties["property_id"] = e_id
    G.save(graph_out)


def build_mini_grpah():
    G = Graph()
    v_label = G.new_vertex_property('string')
    v_id = G.new_vertex_property('int')
    e_id = G.new_edge_property('int')
    e_label = G.new_edge_property('string')

    vertexs = [G.add_vertex() for _ in range(10)]
    edges = [
        G.add_edge(vertexs[0], vertexs[1]),
        G.add_edge(vertexs[1], vertexs[2]),
        G.add_edge(vertexs[2], vertexs[3]),
        G.add_edge(vertexs[4], vertexs[5]),
    ]
    for i, e in enumerate(edges):
        e_id[e] = i
        e_label[e] = f"{i}"

    G.vertex_properties["label"] = v_label
    G.vertex_properties["id"] = v_id
    G.edge_properties["property_label"] = e_label
    G.edge_properties["property_id"] = e_id
    
    graph_out = './mini_test.xml.gz'
    G.save(graph_out)
    Gl = load_graph(graph_out)

    edge = Gl.edge(Gl.vertex(0), Gl.vertex(1))
    tmp = Gl.ep['property_id'][edge]
    print('edge_id: ', tmp)
    tmp = Gl.ep['property_label'][edge]
    print('edge_label: ', tmp)


def check_gtool_graph(graph_file):
    openmp_set_num_threads(16)
    G = load_graph(graph_file)
    print('Graph loaded!')
    
    jewish = find_vertex(G, "id", 7325)
    nazi = find_vertex(G, "id", 574578)
    import pdb; pdb.set_trace()
    print(nazi)


def filter_edge(data_dir):
    assert os.path.exists(data_dir)
    # item_tab = pd.read_csv(os.path.join(data_dir, 'item.csv'))
    statem = pd.read_csv(os.path.join(data_dir, 'statements.csv'))
    prop = pd.read_csv(os.path.join(data_dir, 'property.csv'))
    black_list_id = []
    for i, row in prop.iterrows():
        if ' ID' in row.en_label:
            black_list_id.append(row.property_id)
    print("black_list_id: ", black_list_id)
    import pdb; pdb.set_trace()
    f_state = statem.loc[~statem.edge_property_id.isin(black_list_id)]
    print(f_state)
    print(statem)


def embed_mtx_cos(mtx_a, mtx_b):
    norm_a = np.linalg.norm(mtx_a, keepdims=True, axis=1)
    norm_b = np.linalg.norm(mtx_b, keepdims=True, axis=1)
    mtx_norm = np.matmul(norm_a, norm_b.T)
    mtx_cos = np.matmul(mtx_a, mtx_b.T) / mtx_norm
    return mtx_cos


def create_item_embed(data_dir):
    item_tab = pd.read_csv(os.path.join(data_dir, 'item.csv'))
    aliase_tab = pd.read_csv(os.path.join(data_dir, 'item_aliases.csv'))
    
    # roberta = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    roberta = RobertaModel.from_pretrained('roberta-large')

    with torch.no_grad():
        for i, row in item_tab.iterrows():
            primary_label = row.en_label
            iid = row.item_id
            alias = aliase_tab[aliase_tab.item_id == iid].en_alias.values.tolist()
            # embed = roberta.encode([primary_label] + alias)
            
            encoded_input = tokenizer(
                ['A small cat', primary_label] + alias,
                return_tensors='pt',
                padding=True)
            output = roberta(**encoded_input)
            embed = output[1].cpu().numpy()
            _cos_dis = embed_mtx_cos(embed, embed)
            import pdb; pdb.set_trace()
            break



if __name__ == "__main__":
    # build_statement_nx_graph(
    #     '/media/ron/B008F6C208F6871E/KenshoDerivedWikimediaDataset',
    #     '/media/ron/B008F6C208F6871E/KenshoDerivedWikimediaDataset/wiki_nx.pickle'
    # )
    # build_statement_gtool_graph(
    #     '/media/ron/B008F6C208F6871E/KenshoDerivedWikimediaDataset',
    #     '/media/ron/B008F6C208F6871E/KenshoDerivedWikimediaDataset/wikidata.xml.gz'
    # )
    # check_gtool_graph("/home/ron/Documents/wikidata.xml.gz")
    # filter_edge("/media/ron/B008F6C208F6871E/KenshoDerivedWikimediaDataset")
    # build_mini_grpah()
    create_item_embed('/media/ron/B008F6C208F6871E/KenshoDerivedWikimediaDataset')