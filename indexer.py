import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import math
from collections import defaultdict
import lucene
from collections import Counter
import glob
from java.io import File
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader, Term
from org.apache.lucene.search import IndexSearcher, TermQuery
from org.apache.lucene.util import BytesRefIterator
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute


#Initializa JVM
lucene.initVM()


stopwords_path = "/Users/atharvadashora/Downloads/IR-Assignment-2/nfcorpus/raw/stopwords.large"
indexDir = "/Users/atharvadashora/Downloads/IR-Assignment-2/index"
docIDs = []
with open(stopwords_path) as f:
    stopwords = f.read().split('\n')

#experiment 1

def preProcess(data):
    stemmer = PorterStemmer()
    # data = nltk.word_tokenize(data)
    for word in data:
        word = word.lower()
    return data
#start indexing
def index_doc_dump():
    if not os.path.exists(indexDir):
        os.makedirs(indexDir)

    analyzer = StandardAnalyzer()

    indexdir = File(indexDir).toPath()
    indexPath = FSDirectory.open(indexdir)


    config = IndexWriterConfig(analyzer)
    writer = IndexWriter(indexPath, config)


    input_file = "/Users/atharvadashora/Downloads/IR-Assignment-2/nfcorpus/raw/doc_dump.txt"

    field_filename = FieldType()
    field_filename.setStored(True)
    field_filename.setTokenized(False)
    field_filename.setIndexOptions(IndexOptions.DOCS)
    
    field_filedata = FieldType()
    field_filedata.setStored(True)
    field_filedata.setTokenized(True)
    field_filedata.setStoreTermVectors(True)
    field_filedata.setStoreTermVectorPositions(True)
    field_filedata.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            id, url, title, abstract = line.split('\t')
            doc = Document()
            # docIDs.append(id)
            doc.add(Field("id", id, field_filename))
            doc.add(Field("url", url, field_filename))
            doc.add(Field("title", title, field_filename))
            doc.add(Field("body", preProcess(title+" "+abstract), field_filedata))
            writer.addDocument(doc)

    writer.commit()
    writer.close()

    print("Indexing doc_dump completed successfully.")


def index_nf_dump():
    # indexDir = "/Users/atharvadashora/Downloads/IR-Assignment-2/index"
    if not os.path.exists(indexDir):
        os.makedirs(indexDir)


    analyzer = StandardAnalyzer()
    indexdir = File(indexDir).toPath()
    indexPath = FSDirectory.open(indexdir)


    config = IndexWriterConfig(analyzer)
    writer = IndexWriter(indexPath, config)


    input_file = "/Users/atharvadashora/Downloads/IR-Assignment-2/nfcorpus/raw/nfdump.txt"

    field_filename = FieldType()
    field_filename.setStored(True)
    field_filename.setTokenized(False)
    field_filename.setIndexOptions(IndexOptions.DOCS)
    
    field_filedata = FieldType()
    field_filedata.setStored(True)
    field_filedata.setTokenized(True)
    field_filedata.setStoreTermVectors(True)
    field_filedata.setStoreTermVectorPositions(True)
    field_filedata.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            id, url, title, maintext, comments, topic_tags, description, doctors_note, article_links, question_links, topic_links, video_links, medarticle_links = line.split('\t')
            body = ""
            arr = line.split('\t')
            for i in range(3, 8):
                body = body +" "+ arr[i]
            doc = Document()
            # docIDs.append(id)
            doc.add(Field("id", id, field_filename))
            doc.add(Field("url", url, field_filename))
            doc.add(Field("title", title, field_filename))
            doc.add(Field("body", title+" "+maintext, field_filedata))

            writer.addDocument(doc)

    writer.commit()
    writer.close()

    print("Indexing nf_dump completed successfully.")


# index_doc_dump()
# index_nf_dump()

#experiment 2
#vector space model with tf-idf 

analyzer = StandardAnalyzer()
indexdir = File(indexDir).toPath()
indexPath = FSDirectory.open(indexdir)
index_reader = DirectoryReader.open(indexPath)
searcher = IndexSearcher(index_reader)
num_docs = index_reader.numDocs()

#get a list of docIDs

def create_vocab(index_reader):
    vocabulary = set()
    for doc_id in range(index_reader.numDocs()):
        doc = index_reader.document(doc_id)
        body_field = doc.get("body")
        docIDs.append(doc.get("id"))
        for terms in preProcess(body_field).split():
            vocabulary.add(terms)
    return list(vocabulary)

vocabulary = create_vocab(index_reader)
print("Size of Vocabulary:", len(vocabulary))

def QueryVectorGenerator(query):
    queryTerms = word_tokenize(preProcess(query))
    queryVector = defaultdict(lambda: 0)
    for term in queryTerms:
        queryVector[term] += 1
    for terms in [item for item in vocabulary if item not in queryTerms]:
        queryVector[terms] = 0
    return queryVector

# print(QueryVectorGenerator("why deep fried foods may cause cancer"))

def vectorize_document(doc_id):
    doc = index_reader.document(doc_id)
    body_field = doc.get("body")
    bodyTerms = word_tokenize(preProcess(body_field))
    vector = defaultdict(lambda: 0)
    for terms in bodyTerms:
        vector[terms]+=1
    for terms in [item for item in vocabulary if item not in bodyTerms]:
        vector[terms] = 0
    return vector

def DocsWithTerm(term):
    termQuery = TermQuery(Term('body', term))
    hits = searcher.search(termQuery, num_docs).scoreDocs
    docIds = []
    i = 0
    while i < len(hits):
        docIds.append(hits[i].doc)
        i = i + 1
    return docIds

def get_id(id):
    termQuery = TermQuery(Term('id', id))
    hits = searcher.search(termQuery, num_docs).scoreDocs
    return hits[0].doc


def idf(term):
    df = index_reader.docFreq(Term('body', term))
    return math.log((num_docs + 1) /(df + 1), 10)

def tfidf(idf, wtd):
    return idf * wtd

def VectorToTFIDF(vector):
    for term in vector.keys():
        tf = vector[term]
        vector[term] = tfidf(idf(term), tf)
    return vector


def QueryVectorToTFIDF(queryVector):
    for term, tf in queryVector.items():
        queryVector[term] = tfidf(idf(term), tf)
    return queryVector


def computeCosinentc(vector1, vector2):
    v1norm = 0
    for values in vector1.values():
        v1norm += values**2
    v2norm = 0
    for values in vector2.values():
        v2norm += values**2
    print(v1norm, v2norm)
    v1norm = v1norm**(1/2)
    v2norm = v2norm**(1/2)
    dot = 0
    for term in vector1.keys():
        dot += vector1[term]*vector2[term]
    cs = dot/(v1norm*v2norm)
    return cs

def computeCosinennn(vector1, vector2):
    v1norm = 0
    for values in vector1.values():
        v1norm += values**2
    v2norm = 0
    for values in vector2.values():
        v2norm += values**2
    print(v1norm, v2norm)
    v1norm = v1norm**(1/2)
    v2norm = v2norm**(1/2)
    dot = 0
    for term in vector1.keys():
        if vector1[term]*vector2[term] != 0:
            print(term, vector1[term], vector2[term])
            dot += vector1[term]*vector2[term]
    cs = dot
    return cs


def TopKRetrievalnnn(query, k):
    queryVector = QueryVectorGenerator(query)
    docIds = []
    docVectors = {}
    computedSimilarity = []
    topKDocInformation = {}
    
    # Get Documents with the Term
    for term in queryVector.keys():
        docIds += DocsWithTerm(term)

    # Create Vector for Documents
    for docId in docIds:
        docVectors[docId] = vectorize_document(docId)

    # Compute Cosine Similarity Scores
    for id, vector in docVectors.items():
        computedSimilarity.append({"DocId": id, "CosineSimilarityScore": computeCosinennn(queryVector, vector)})
    computedSimilarity = sorted(computedSimilarity, key=lambda docIndex: docIndex["CosineSimilarityScore"],reverse=True)

    # Print Top-K RetrievedDocuments
    i = 0
    while i < k and k < len(computedSimilarity):
        DocName = searcher.doc(computedSimilarity[i]["DocId"]).get('title')
        DocID = searcher.doc(computedSimilarity[i]["DocId"]).get('id')
        print(DocName, DocID)
        topKDocInformation[computedSimilarity[i]["DocId"]] = docVectors[computedSimilarity[i]["DocId"]]
        i += 1
    return topKDocInformation


def TopKRetrievalntc(query, k):
    queryVector = QueryVectorGenerator(query)
    docIds = []
    docVectors = {}
    computedSimilarity = []
    topKDocInformation = {}
    
    # Get Documents with the Term
    for term in queryVector.keys():
        docIds += DocsWithTerm(term)

    # Create Vector for Documents
    for docId in docIds:
        docVectors[docId] = vectorize_document(docId)

    # Transform Document Vectors to TF-IDF Values
    for docId, vector in docVectors.items():
        docVectors[docId] = VectorToTFIDF(vector)

    # Transform Query Vector to TF-IDF Values
    queryVector = QueryVectorToTFIDF(queryVector)

    # Compute Cosine Similarity Scores
    for id, vector in docVectors.items():
        computedSimilarity.append({"DocId": id, "CosineSimilarityScore": computeCosinentc(queryVector, vector)})
    computedSimilarity = sorted(computedSimilarity, key=lambda docIndex: docIndex["CosineSimilarityScore"],reverse=True)

    # Print Top-K RetrievedDocuments
    i = 0
    while i < k and k < len(computedSimilarity):
        DocName = searcher.doc(computedSimilarity[i]["DocId"]).get('title')
        DocID = searcher.doc(computedSimilarity[i]["DocId"]).get('id')
        print(DocName, DocID)
        topKDocInformation[computedSimilarity[i]["DocId"]] = docVectors[computedSimilarity[i]["DocId"]]
        i += 1
    return topKDocInformation


#experiment 3 rocchio
#reading the query file
queries = defaultdict()
directory = '/Users/atharvadashora/Downloads/IR-Assignment-2/nfcorpus'
file_type = '*.queries'

file_list = glob.glob(os.path.join(directory, file_type))
rel_list = {}
# print(file_list)
for file_path in file_list:
    with open(file_path, 'r') as file:
        for line in file:
            # print(line.split())
            qid, query = line.split('\t')
            queries[qid] = query
            rel_list[qid] = []

#reading the qrel file
queryrel = {}
with open('nfcorpus/merged.qrel', 'r') as file:
    for line in file:
        qid, zero, docid, rel = line.split('\t')
        # print(qid, zero, docid, rel)
        qdid = qid+" "+docid
        queryrel[qdid] = rel
        rel_list[qid].append(docid)
docVecAll = {}
# for doc in docIDs:
#     print(doc)
#     docVecAll[doc] = vectorize_document(get_id(doc))
# # rocchio algorithm for queries

def rocchio(qid):
    alpha = 0.75
    beta = 0.5
    gamma = 0.5
    print(qid)
    num_rel = len(rel_list[qid])
    num_non_rel = num_docs - num_rel
    print(num_rel)
    rel_vec = defaultdict(lambda:0)
    for doc in rel_list[qid]:
        # print(get_id(doc))
        # print(type(doc))
        doc_vec = docVecAll[doc]
        for term in doc_vec.keys():
            # qdid = qid+" "+doc
            # print(queryrel[qid+" "+doc])
            rel_vec[term]+=(int(queryrel[qid+" "+doc])*doc_vec[term]/num_rel)
    non_rel_vec = defaultdict(lambda:0)
    for doc in [item for item in docIDs if item not in rel_list[qid]]:
        doc_vec = docVecAll[doc]
        print(doc)
        for term in doc_vec.keys():
            non_rel_vec[term]+=(doc_vec[term]/num_non_rel)
    
    query_vec = QueryVectorGenerator(queries[qid])
    # print(query_vec)
    for i in range(3):
        for term in query_vec.keys():
            query_vec[term]*=alpha
        for term in query_vec.keys():
            query_vec[term]+=(beta*rel_vec[term]-gamma*non_rel_vec[term])
    
    return query_vec

# rocchio(list(queries.keys())[0])


#experiment 4

#probabilostic 
#language model

def find_prob(qid, doc):
    f = 0.5
    prob = 1
    doc_vec = docVecAll[doc]
    doc_size = 0
    for val in doc_vec.values():
        doc_size+=val
    # print(doc_size)
    # print(doc)
    for terms in (preProcess(queries[qid]).split()):
        # print(terms)
        total_term_freq = index_reader.totalTermFreq(Term('body', terms))
        # df = index_reader.docFreq(Term('body', terms))
        # print(df)
        # print(total_term_freq)
        # print(doc_vec[terms])
        prob*=(f*(doc_vec[terms]/doc_size)+(1-f)*(total_term_freq/len(vocabulary)))
    # print(prob)
    return prob

def language_model_rank(qid):
    rank_dict = {}
    for doc in docIDs:
        rank_dict[qid+" "+doc] = find_prob(qid, doc)
    rank_dict = {k: v for k, v in sorted(rank_dict.items(), key=lambda item: (-1)*item[1])}
    return rank_dict




#bm-25

def get_av_doclen():
    s = 0
    for doc in docIDs:
        s += sum(list(docVecAll[doc].values()))
    s/=num_docs
    return s


def find_rsv(qid, doc):
    k1 = 1.5
    k3 = 1.5
    b = 0.75
    rsv = 0
    ld = 0
    for val in docVecAll[doc].values():
        ld+=val

    lav = get_av_doclen()
    query_vec = QueryVectorGenerator(qid)
    for terms in (word_tokenize(preProcess(queries[qid]))):
        tf_q = query_vec[terms]
        tf_d = docVecAll[doc][terms]
        df = index_reader.docFreq(Term('body', terms))
        rsv+=(math.log(df/num_docs)*((k1+1)*tf_d/(k1*(1-b+b*(ld/lav))+tf_d))*(k3+1)*tf_q/(k3+tf_q))

    return rsv




##implement rocchio with relevance feedback
##run whole thing once


#exp 5
import pandas as pd

df = pd.read_csv('/Users/atharvadashora/Downloads/IR-Assignment-2/gena_data_final_triples.csv')

entity_set = set(df['Subject'].unique()) | set(df['Object'].unique())
def entity_based_retrieval():
    f_corr = {}
    f_ef = {}
    for qid in list(queries.keys()):
        query = queries[qid]
        print(1)
        f_corr_list = {}
        f_ef_list = {}
        for doc_id in docIDs:
            f_c = 0
            f_e = 0
            doc = index_reader.document(get_id(doc_id))
            body_field = doc.get("body")
            for entity in entity_set:
                if str(entity) in query and str(entity) in body_field:
                    f_c+=1
                    f_e+=(query.count(entity)*math.log(1+body_field.count(entity)))
            f_corr_list[doc_id] = f_c
            f_ef_list[doc_id] = f_e
            print(f_c, f_e)
        f_corr[qid] = f_corr_list
        f_ef[qid] = f_ef_list
    return f_corr, f_ef


#exp 6

#query expansion using knowledge graph

# Load triples from CSV into a dictionary
import csv

def load_triples_from_csv(csv_file):
    triples = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 3:
                subject, relation, obj = row
                if subject not in triples:
                    triples[subject] = []
                triples[subject].append((relation, obj))
    return triples

def expand_query(query, triples):

    entities = []
    for entity in entity_set:
        if str(entity) in query:
            entities.append(entity)
    
    expanded_query = set(entities)  
    
    for entity in entities:
        if entity in triples:
            related_entities = [triple[1] for triple in triples[entity]]
            expanded_query.update(related_entities)
    
    return expanded_query


triples = load_triples_from_csv('/Users/atharvadashora/Downloads/IR-Assignment-2/gena_data_final_triples.csv')

# print(len(expand_query("obesity and enxiety", triples)))
# print(len(entity_set))

#learning to rank

#here we need to use tensorflow in randem with the relevance feedback we have to implement some learning to rank models

