# Text2Pred
Open information extraction using the Stanford parser - using the StanfordCoreNLP wrapper. 

INPUT: 
"John drove his new car but he crashed it."

OUTPUT: 
[['John',    'drove',   'car'], 
 ['John_he', 'crashed', 'his_new_car_it']


The resulting predicates overlap to form a document knowledge graph, in the forms of Subject _Relation Object triples_. Nodes store nouno-based informaiton while edges typically hold verb labels. 
The output typically forms a directed graph with multi edges, including some self loops. 
