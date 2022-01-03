import logging
import sys
from AlignmentFormat import serialize_mapping_to_tmp_file
from collections import defaultdict
import numpy as np
import random
import json
from Config import Batch,Config
from Trans import Trans
import os
#from rdflib import Graph, URIRef, RDFS
from bs4 import BeautifulSoup
from owlready2 import onto_path, get_ontology 
import io
from time import sleep
import itertools
from tqdm import tqdm
from fuzzywuzzy import fuzz,process
import urllib.request

#Pre-processing the source, target and reference files
# def render_using_label(entity):
#     return entity.label.first() or entity.name

def read_ontology(file):
    onto = get_ontology(file)
    onto.load()
    base = onto.base_iri

    # Read classes
    entity_list = {}
    label_list = {}
    id_node = 0
    #excl = ['DbXref','Definition','ObsoleteClass','Subset','Synonym','SynonymType']

    for cl in onto.classes(): 
        if cl not in entity_list:
            #if cl.name not in excl:
            entity_list[base+cl.name] = id_node            
            labels = cl.label
            if len(labels)==0:
                label = cl.name
                label = label.lower().replace('_',' ')
                label = label.replace('/',' ')
                label = label.replace("-",' ')
                label = label.replace("of",'')
                label = label.replace("this",'')
                label = label.replace("the",'')
                label = label.replace("or",'')
                label = label.replace("and",'')
                label = label.replace("that",' ')
                label = label.replace("by",' ')
                label = label.replace(" ",'_')
                label = label.replace("_a_",'_')
                label = label.replace("_an_",'_')
                label = label.replace("_",' ')
                label_list[label] = id_node 
            else:
                for label in labels:
                    label = label.lower().replace('_',' ')
                    label = label.replace('/',' ')
                    label = label.replace("-",' ')
                    label = label.replace("of",'')
                    label = label.replace("this",'')
                    label = label.replace("the",'')
                    label = label.replace("or",'')
                    label = label.replace("and",'')
                    label = label.replace("that",' ')
                    label = label.replace("by",' ')
                    label = label.replace(" ",'_')
                    label = label.replace("_a_",'_')
                    label = label.replace("_an_",'_')
                    label = label.replace("_",' ')
                    label_list[label] = id_node          
        id_node = id_node+1
    
    return entity_list,label_list

def getSubclass(file):
    subclass = []
    onto = get_ontology(file)
    onto.load()
    base = onto.base_iri
    
    classes = []
    excl = ['Thing','DbXref','Definition','ObsoleteClass','Subset','Synonym','SynonymType']

    for cl in onto.classes():
        #if cl.name not in excl:
        classes.append(cl)

    classes = list(set(classes))

    for i in range (len(classes)):
        clss = base+classes[i].name
        for j in range(len(classes[i].is_a)):
            try:
                if classes[i].is_a[j].name not in excl:
                    subclass.append((clss,base+classes[i].is_a[j].name))
            except AttributeError:
                pass
    return subclass

def getDijclass(file):
    dijclass = []
    onto = get_ontology(file)
    onto.load()
    base = onto.base_iri
    classes = []
    #excl = ['DbXref','Definition','ObsoleteClass','Subset','Synonym','SynonymType']

    for cl in onto.classes():
        #if cl.name not in excl:
        classes.append(base+cl.name)

    classes = list(set(classes))

    try:
        for clss in classes:
            for d in clss.disjoints():
                dijclass.append(d.entities)
    except AttributeError:
        pass
    return dijclass

def getTriples(file):
    triples = []
    djw_list = {}
    subclass = getSubclass(file)
    dijclass = getDijclass(file)
    classes,labels = read_ontology(file)

    excl = ['Thing','DbXref','Definition','ObsoleteClass','Subset','Synonym','SynonymType']
    
    sb_list = {}
    for pair in subclass:
        if pair[0] not in sb_list:
            sb_list[pair[0]]=classes.get(pair[0])
        if pair[1] not in sb_list:
            sb_list[pair[1]]=classes.get(pair[1])

    for sub in subclass:
        triples.append([classes[sub[0]],1,classes[sub[1]]])

    for dij in dijclass:
        triples.append([classes.get(dij[0].name),2, classes.get(dij[1].name)])

    diffKeys = set(classes.keys()) - set(sb_list.keys())
    for key in diffKeys:
        djw_list[key] = classes.get(key)
    for entity1, id1 in sb_list.items():
        for entity2, id2 in djw_list.items():
            triples.append([sb_list[entity1] , 2, djw_list[entity2]])
    
    return triples

def getSyn(file):
    
    entity_list,label_list = read_ontology(file)
    onto = get_ontology(file)
    onto.load()
    base = onto.base_iri
    
    synonym_list = {}

    with open(file) as f:
        soup = BeautifulSoup(f,'xml')
    try:
        cells = soup.find_all('owl:Class')
        for cell in tqdm(cells):
            entity = cell.attrs['rdf:about']
            #entity = entity.replace(base,'')
            if entity in entity_list.keys():        
                if cell.find('oboInOwl:hasRelatedSynonym') is not None:
                    for synonyms in cell.findAll('oboInOwl:hasRelatedSynonym'):
                        synonym = synonyms['rdf:resource']
                        synonym_list[synonym] = entity_list.get(entity)
    except (KeyError,AttributeError):
        pass


    with open(file) as f:
        soup = BeautifulSoup(f,'xml')

    try:        
        cells_synonym = soup.find_all('rdf:Description')
        for cell in cells_synonym:
            synonym = cell.attrs['rdf:about']
            if synonym in synonym_list.keys():
                if cell.find('rdfs:label') is not None:
                    synonym_label = cell.find('rdfs:label').string.lower().replace('_',' ')
                    synonym_label = synonym_label.replace('/',' ')
                    synonym_label = synonym_label.replace("-",' ')
                    synonym_label = synonym_label.replace(",",' ')
                    synonym_label = synonym_label.replace("of",'')
                    synonym_label = synonym_label.replace("this",'')
                    synonym_label = synonym_label.replace("the",'')
                    synonym_label = synonym_label.replace("or",'')
                    synonym_label = synonym_label.replace("and",'')
                    synonym_label = synonym_label.replace("that",'')
                    synonym_label = synonym_label.replace(" ",'_')
                    synonym_label = synonym_label.replace("_a_",'_')
                    ynonym_label = synonym_label.replace("_an_",'_')
                    synonym_label = synonym_label.replace("_",' ')
                    if synonym_label not in label_list:
                        label_list[synonym_label]=synonym_list.get(synonym)
    except (KeyError,AttributeError):
        pass
    
    return label_list  
    

def getTrainfileS(source_graph):
    
    classes_source,labels_source = read_ontology(source_graph)
    labels_source_syn = getSyn(source_graph)
    triples_source = getTriples(source_graph)

    entity2id_file = open("ent_ids_source.txt", "w")
    for entity , id in classes_source.items():
        entity = entity.encode('utf8').decode('utf-8')
        entity2id_file.write(str(id) + '\t' + str(entity)+ '\n')
    entity2id_file.close()

    label2id_file = open("label_ids_source.txt", "w")
    for label , id in labels_source_syn.items():
        label = label.encode('utf8').decode('utf-8')
        label2id_file.write(str(id) + '\t' + str(label)+ '\n')
    label2id_file.close()
    
    train2id_file = open("triples_source.txt", "w")
    for triple in triples_source:
        train2id_file.write(str(triple[0]) + '\t' + str(triple[2]) + '\t' + str(triple[1]) + '\n')
    train2id_file.close()


def getTrainfileT(target_graph):
    
    classes_target,labels_target = read_ontology(target_graph)
    labels_target_syn = getSyn(target_graph)
    triples_target = getTriples(target_graph)

    entity2id_file = open("ent_ids_target.txt", "w")
    for entity , id in classes_target.items():
        entity = entity.encode('utf8').decode('utf-8')
        entity2id_file.write(str(id) + '\t' + str(entity)+ '\n')
    entity2id_file.close()

    label2id_file = open("label_ids_target.txt", "w")
    for label , id in labels_target_syn.items():
        label = label.encode('utf8').decode('utf-8')
        label2id_file.write(str(id) + '\t' + str(label)+ '\n')
    label2id_file.close()

    train2id_file = open("triples_target.txt", "w")
    for triple in triples_target:
        train2id_file.write(str(triple[0]) + '\t' + str(triple[2]) + '\t' + str(triple[1]) + '\n')
    train2id_file.close()

def getTrainList(source_graph,target_graph):
    train_list = []
    labels_source_syn = getSyn(source_graph)
    labels_target_syn = getSyn(target_graph)

    if len(labels_source_syn)==0 or len(labels_target_syn)==0:
        labels_source_syn,label_list = read_ontology(source_graph)
        labels_target_syn,label_list = read_ontology(target_graph)

    for k in sorted(labels_source_syn.keys() & labels_target_syn.keys()):
        train_list.append([labels_source_syn[k],labels_target_syn[k]])

    label_list =[]
    labels = list(itertools.product(list(labels_source_syn.keys()[:6000]), list(labels_target_syn.keys()[:6000])))
    for a, b in tqdm(labels):
        label_list.append([a,b])

    for i in tqdm(range(len(label_list))):
        strings1 = label_list[i][0]
        string1 = strings1.replace('_',' ')
        strings2 = label_list[i][1]
        string2 = strings2.replace('_',' ')
        fuzzs = fuzz.token_sort_ratio(string1,string2)
        a  = labels_source_syn.get(str(strings1))
        b  = labels_target_syn.get(str(strings2))
        if fuzzs >= 97:
            if [a,b] not in train_list:
                train_list.append([a,b])
        # if len(train_list) <=10:
        #     if fuzzsp >=95:
        #         if [a,b] not in train_list:
        #             train_list.append([a,b])
        else:
            pass


    return train_list

def getFile(source_url,target_url):
    getTrainfileS(source_url)
    getTrainfileT(target_url)

    train_list = getTrainList(source_url,target_url)
    train2id_file = open("train.txt", "w")
    for triple in train_list:
        train2id_file.write(str(triple[0]) + '\t' + str(triple[1]) + '\n')
    train2id_file.close()

    #generate negative samples
    negitive_sampling_constrain_source = {}
    sbpt_source = {}
    dis_source = {}
    f_source = open('triples_source.txt', 'r')
    train2id_source = []

    for line in f_source.readlines()[1:]:
        train2id_source.append(line)
    f_source.close()

    for line in train2id_source:
        train2id = line.strip('\n').split('\t')
        leftent, rightent, rel = int(train2id[0]), int(train2id[1]), int(train2id[2])
        #if "subClassOf" in rel_source:
        if rel == 1:
            if str(leftent) not in sbpt_source.keys():
                sbpt_source[str(leftent)] = []
            if rightent not in sbpt_source[str(leftent)]:
                sbpt_source[str(leftent)].append(rightent)
            if str(rightent) not in sbpt_source.keys():
                sbpt_source[str(rightent)] = []
            if leftent not in sbpt_source[str(rightent)]:
                sbpt_source[str(rightent)].append(leftent)
            else:
                pass
        else:
            pass

        #if "disjointWith" in rel_source:
        if rel == 2:
            if str(leftent) not in dis_source.keys():
                dis_source[str(leftent)] = []
            if rightent not in dis_source[str(leftent)]:
                dis_source[str(leftent)].append(rightent)
            if str(rightent) not in dis_source.keys():
                dis_source[str(rightent)] = []
            if leftent not in dis_source[str(rightent)]:
                dis_source[str(rightent)].append(leftent)
            else:
                pass
        else:
            pass
    negitive_sampling_constrain_source['sbc'] = sbpt_source
    negitive_sampling_constrain_source['dij'] = dis_source

    f = open('neg_constrain_source.json', 'w')
    json.dump(negitive_sampling_constrain_source, f)
    f.close()

    negitive_sampling_constrain_target = {}
    sbpt_target = {}
    dis_target = {}
    f_target = open('triples_target.txt', 'r')
    train2id_target = []
    for line in f_target.readlines()[1:]:
        train2id_target.append(line)
    f_target.close()
    for line in train2id_target:
        train2id = line.strip('\n').split('\t')
        leftent, rightent, rel = int(train2id[0]), int(train2id[1]), int(train2id[2])
        #if "subClassOf" in rel_source:
        if rel == 1:
            if str(leftent) not in sbpt_target.keys():
                sbpt_target[str(leftent)] = []
            if rightent not in sbpt_target[str(leftent)]:
                sbpt_target[str(leftent)].append(rightent)
            if str(rightent) not in sbpt_target.keys():
                sbpt_target[str(rightent)] = []
            if leftent not in sbpt_target[str(rightent)]:
                sbpt_target[str(rightent)].append(leftent)
            else:
                pass
        else:
            pass

        #if "disjointWith" in rel_target:
        if rel == 2:
            if str(leftent) not in dis_target.keys():
                dis_target[str(leftent)] = []
            if rightent not in dis_target[str(leftent)]:
                dis_target[str(leftent)].append(rightent)
            if str(rightent) not in dis_target.keys():
                dis_target[str(rightent)] = []
            if leftent not in dis_target[str(rightent)]:
                dis_target[str(rightent)].append(leftent)
            else:
                pass
        else:
            pass
    negitive_sampling_constrain_target['sbc'] = sbpt_target
    negitive_sampling_constrain_target['dij'] = dis_target

    f = open('neg_constrain_target.json', 'w')
    json.dump(negitive_sampling_constrain_target, f)
    f.close()

def embedding_run():
    #Getting translation based embeddings
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    con = Config()
    con.inPath("")
    con.trainTimes(1000)
    con.setBatches(100)
    con.learningRate(0.01)
    con.entDimension(100)
    con.negRate(10)
    con.negativeSampling('unif')
    con.optMethod("SGD")
    con.exportFiles("amd.model.tf", 0)
    con.vecFiles("amd.embedding.vec.json")
    con.model_name("trans")
    con.init()
    con.model(Trans)
    con.run()


#Begin matching
def getID():    
    with open('ent_ids_source.txt') as f:
        ent_ids_source = {line.strip().split('\t')[0]: int(line.strip().split('\t')[0]) for line in f.readlines()}
    with open('ent_ids_target.txt') as f:
        ent_ids_target = {line.strip().split('\t')[0]: int(line.strip().split('\t')[0]) for line in f.readlines()}

    f = open("amd.embedding.vec.json", "r")
    embedding = json.load(f)
    f.close()
    source_vecs = embedding["source_ent_embeddings"][:len(ent_ids_source)+1]
    target_vecs = embedding["target_ent_embeddings"][:len(ent_ids_target)+1]

    return ent_ids_source,ent_ids_target,source_vecs,target_vecs

def getList():
    source_list = []
    target_list = []
    with open('ent_ids_source.txt') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            entid = int(params[0])
            ent = str(params[1])
            source_list.append(ent)
        f.close()

    with open('ent_ids_target.txt') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            entid = int(params[0])
            ent = str(params[1])
            target_list.append(ent)
        f.close()

    return source_list,target_list

def getAligns(soure_list, target_list, threshold):
    sim_dict = {}
    vec_alignments = []
    ent_ids_source,ent_ids_target,source_vecs,target_vecs = getID()
    for ent1 in tqdm(soure_list):
        source_vec = source_vecs[soure_list.index(ent1)]
        for ent2 in target_list:
            target_vec = target_vecs[target_list.index(ent2)]
            simility = np.dot(target_vec, source_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(source_vec))
            sim_dict[ent1 + '\t' + ent2] = simility
            if simility >= threshold: 
                vec_alignments.append((ent1, ent2))
    return sim_dict, vec_alignments

def alignmentMatch(source_url, target_url):
    relation = '='
    alignments = []
    train = []
    aligns = []
    ent12ent2 = {}
    getFile(source_url, target_url)
    embedding_run()
    threshold = 0.95  
    ent_ids_source,ent_ids_target,source_vecs,target_vecs = getID()
    source_list,target_list = getList()
    sim_dict, vec_alignments = getAligns(source_list, target_list, threshold)

    with open('ent_ids_source.txt','r') as f:
        sourceent2id = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f.readlines()}
    with open('ent_ids_target.txt','r') as f:
        targetent2id = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f.readlines()}
    
    with open('train.txt','r') as f:
        for line in f.readlines():
            pair = line.strip().split('\t')
            train.append((int(pair[0]), int(pair[1])))
        f.close()  
    
    for align in tqdm(vec_alignments):
        source, target = align[0], align[1]
        simility = sim_dict[source + '\t' + target]
        aligns.append((source,target))
        alignments.append((source,target,relation,round(simility)))

    for i in range(len(train)):
        a = sourceent2id.get(str(train[i][0]))
        b = targetent2id.get(str(train[i][1]))
        if (a,b) not in aligns:
            alignments.append((a,b,relation,1.0))  
    #print(alignments)

    return alignments

def match_rdflib(source_graph, target_graph, input_alignment):
    # a very simple label matcher:
    alignment = []

    label_to_uri = defaultdict(list)
    for s, p, o in source_graph.triples((None, RDFS.label, None)):
        if isinstance(s, URIRef):
            label_to_uri[str(o)].append(str(s))

    for s, p, o in target_graph.triples((None, RDFS.label, None)):
        if isinstance(s, URIRef) and str(o) in label_to_uri:
            for one_uri in label_to_uri[str(o)]:
                alignment.append((one_uri, str(s), "=", 1.0))
    return alignment
    # return [('http://one.de', 'http://two.de', '=', 1.0)]


def get_file_from_url(location):
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname, urlopen

    if location.startswith("file:"):
        return open(url2pathname(unquote(urlparse(location).path)))
    else:
        return urlopen(location)


def match(source_url, target_url, input_alignment_url):
    logging.info("Python matcher info: Match " + source_url + " to " + target_url)

    urllib.request.urlretrieve(source_url, "source.owl")
    urllib.request.urlretrieve(target_url, "target.owl")
    source_file = "source.owl"
    target_file = "target.owl"

    # onto_source = get_ontology(source_url)
    # onto_source.load
    # base_source = onto_source.base_iri

    # onto_target = get_ontology(target_url)
    # onto_target.load
    # base_target = onto_target.base_iri

    # in case you want the file object use
    #source_file = get_file_from_url(source_url)
    #target_file = get_file_from_url(target_url)

    #source_graph = Graph()
    #source_graph.parse(source_url)
    #logging.info("Read source with %s triples.", len(source_graph))

    #target_graph = Graph()
    #target_graph.parse(target_url)
    #logging.info("Read target with %s triples.", len(target_graph))

    #input_alignment = None
    # if input_alignment_url is not None:

    resulting_alignment = alignmentMatch(source_file, target_file)

    # in case you have the results in a pandas dataframe, make sure you have the columns
    # source (uri), target (uri), relation (usually the string '='), confidence (floating point number)
    # in case relation or confidence is missing use: df["relation"] = '='  and  df["confidence"] = 1.0
    # then select the columns in the right order (like df[['source', 'target', 'relation', 'confidence']])
    # because serialize_mapping_to_tmp_file expects an iterbale of source, target, relation, confidence
    # and then call .itertuples(index=False)
    # example: alignment_file_url = serialize_mapping_to_tmp_file(df[['source', 'target', 'relation', 'confidence']].itertuples(index=False))

    alignment_file_url = serialize_mapping_to_tmp_file(resulting_alignment)
    return alignment_file_url


def main(argv):
    if len(argv) == 2:
        print(match(argv[0], argv[1], None))
    elif len(argv) >= 3:
        if len(argv) > 3:
            logging.error("Too many parameters but we will ignore them.")
        print(match(argv[0], argv[1], argv[2]))
    else:
        logging.error(
            "Too few parameters. Need at least two (source and target URL of ontologies"
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO
    )
    main(sys.argv[1:])
