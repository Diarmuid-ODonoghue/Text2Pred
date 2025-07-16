# Text2Pred
Open information extraction using the Stanford parser - using the **StanfordCoreNLP** wrapper. 

INPUT: 
"John drove his new car but he crashed it."

OUTPUT:  
|  Subject | Relation | Object |
| ------ | ------ | ------ |
| John | drove | car |
| John_he  | crashed  |  his_new_car_it |

Note that coreferences have been included forming the concept node 'his_new_car_it' as multiple words refer to the same concept. Predicates are in the form of **(Subject Relation Object)** triples and these combine to form a document knowledge graph. Nodes store noun-based information while edges typically hold verb labels. The output generally forms a directed graph with multi edges, including some self loops. 
