import os
from database_manager import ResearchDB
print('GEMINI_API_KEY present:', bool(os.getenv('GEMINI_API_KEY')))

db = ResearchDB(persist_directory='debug_db', collection_name='debug_pubmed')
print('DB enabled:', db.enabled)
print('Collection object:', db.collection)

abstracts = [
    "TITLE: Bone Scaffolds\nABSTRACT: Study on biocompatible bone scaffolds for orthopedic applications",
    "TITLE: Dental Implants\nABSTRACT: Analysis of titanium implants and osseointegration"
]
metadatas = [{"year":"2020","link":"http://example.com/1"},{"year":"2021","link":"http://example.com/2"}]
ids = ['debug_pmid_1','debug_pmid_2']

add_res = db.add_abstracts(abstracts, metadatas, ids)
print('add result:', add_res)

try:
    get_res = db.collection.get(ids=ids, include=['documents','metadatas','ids'])
    print('collection.get:', get_res)
except Exception as e:
    print('collection.get error:', e)

try:
    q_res = db.collection.query(query_texts=['bone implants'], n_results=2, include=['documents','metadatas','ids','distances'])
    print('collection.query:', q_res)
except Exception as e:
    print('collection.query error:', e)
