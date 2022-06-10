# Text2Pred
Open information extraction using Stanford parser - in Python using StanfordCoreNLP. 

INPUT: "John drove his new car but he crashed it."

OUTPUT: [['John', 'drove', 'car'], ['John_he', 'crashed', 'his_new_car_it']

The resulting predicates overlap to a document knowledge graph, stored in NetworkX format. Node attributes store conept informaiton while edge attributes store relational informaion. 
