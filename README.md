# Text2Pred
Open information extraction using Stanford parser - in Python using StanfordCoreNLP wrapper. 

INPUT: 
"John drove his new car but he crashed it."

OUTPUT: 
[['John',    'drove',   'car'], 
 ['John_he', 'crashed', 'his_new_car_it']


The resulting predicates overlap to a document knowledge graph, stored in NetworkX format. Node attributes store concept informaiton while edge attributes store relational information. 
The output triples can be used to generate a NetworkX.MultiDiGraph()
