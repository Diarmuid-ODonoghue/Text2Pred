#############################
########## Text2Pred ########
#############################

import csv
import os
from os import path
import pprint
import re
import json
import sys
import nltk

from nltk.tokenize import sent_tokenize
from nltk.tree import *
from functions import *
#import nltk
from pycorenlp import StanfordCoreNLP

from datasets import load_dataset
# dataset = load_dataset("docred")

java_path = "C:\Program Files\Java\jdk1.8.0_171\bin\java.exe"
os.environ['JAVAHOME'] = java_path
# base_path = "C:/Users/dodonoghue/Documents/Python-Me/data/"
nlp = StanfordCoreNLP("http://localhost:9000")

base_path = dir_path = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
local_path = base_path.replace("Cre8Blend", "data", 1)
localBranch = "/test/"# "test/"  #"iProvaData/"
#localBranch = "/Covid-19 on Feb 21/"  # Covid-19 on Feb 21
# localBranch = "MisTranslation data/"
localBranch = "/Psychology data/"
# localBranch = "/Aesops Fables/"
#localBranch = "Covid-19 Publications Feb 21/covid_text/"
# localBranch = "/Sheffield-Plagiarism-Corpus/"
#localBranch = "Killians Summaries/"
#localBranch = "/20 SIGGRAPH Abstracts - Stanford/"
#in_path = localPath + "SIGGRAPH Abstracts/" # "
#in_path = localPath + "psych texts/"
in_path = local_path + localBranch  # "test/"
out_path = local_path + localBranch


###import stanza
###from stanza.server import CoreNLPClient
###stanza.download('en')
###nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')
pp = pprint.PrettyPrinter(indent=4)

global document_triples
global sentence_triples
global sentence_triplesPP 
global set_of_raw_concepts
global sentence_number
global skip_over_previous_results

skip_over_previous_results = False # True # dont regenerate existing outputs

data = []
document_triples = []
sentence_triples = []
sentence_triplesPP = []
fullDocument = ""
sentence_number = 0



concept_tags = {'NN', 'NNS', 'PRP', 'PRP$', 'NNP', 'NNPS'}  # NNP, NNPS
relation_tags = {'VB', 'VBG', 'VBG', 'VBN', 'VBP', 'VBZ', 'VBD'}
illegal_concept_nodes = {"-RRB-", "-LRB-", "-RSB-" "-LSB-" "Unknown", "UNKNOWN", ",", ".", "?", "'s", "'", "''"}

def trim_concept_chain(text): # very long chains only, phrases
    "Extract nouns and Preps from coreferents that are entire phrases."
    str = nltk.word_tokenize(text.replace("_", " "))
    #print(" !!", str, end="!! ")
    tagged = nltk.pos_tag(str)
    ret = '_'.join([word for word, tag in tagged[:-1] if tag in concept_tags] + [tagged[-1][0]])
    return ret
# trim_concept_chain('cloth_captured_John_Americans_from_a_flapping_flag_it')


def processDocument(text): # full document
    global sentence_number
    global sentence_triples
    global sentence_triplesPP
    global set_of_raw_concepts
    list_of_sentences = sent_tokenize(text)
    sents = sent_tokenize(text)
    output = nlp.annotate(text, properties={
                  'annotators': 'tokenize, ssplit, parse, ner, dcoref', 'outputFormat': 'json'})
    #    'annotators': 'tokenize, ssplit, parse, coref', 'outputFormat': 'json'  })
    #output = nlp.annotate(text, properties={
    #     'annotators': 'tokenize, ssplit, parse, ner, coref',  # NEURAL coref
    #     'coref.algorithm': 'neural', 'outputFormat': 'json'  })
    #output = nlp.annotate(text, properties={
    #     'annotators': 'tokenize, ssplit, pos, nerparse, lemma, coref',
    #     'coref.algorithm': 'neural', 'outputFormat': 'json'  })
    if output == "CoreNLP request timed out. Your document may be too long.":
        print("** Timeout of the Stanford Parser")
        list_of_sentences = []  # No parsed output to process
    elif isinstance(output, dict):
        try:
            coref = output['corefs']  ## OCCASIONALY DODGY RESULTS - ??? finds no corefs????
        except IndexError:
            coref = None
    elif type(output) is str:# or type(output) is unicode:
        output = json.loads(output, encoding='utf-8', strict=False)
        try:
            coref = output['corefs']
        except IndexError:
            coref = None
    else:
        print("** Stanford Parser Error - type:", type(output), end="")

    # record multi-word NERs for each sentence, incorporate in subject and object as appropriate.
    ner_dict = {}
    print("NER: ", end="")
    for coref_chain in output['corefs']:
        if output['corefs'][coref_chain][0]['type'] == 'PROPER' and ' ' in output['corefs'][coref_chain][0]['text']:
            if output['corefs'][coref_chain][0]['sentNum'] -1 in ner_dict.keys():
                ner_dict[output['corefs'][coref_chain][0]['sentNum'] -1].append(output['corefs'][coref_chain][0]['text'])
            else:
                ner_dict[output['corefs'][coref_chain][0]['sentNum'] -1] \
                    = [output['corefs'][coref_chain][0]['text']]

    for i in range(len(list_of_sentences)):
        sentence_triples = []
        sentence_triplesPP = []
        sentence_number += 1
        if i in ner_dict.keys():
            sentence_ners = ner_dict[i]
        else:
            sentence_ners = []
        try:
            sent1 = output['sentences'][i]['parse']
        except IndexError:
            sent1 = None
        if sent1 is not None:
            try:
                sent2 = CoreferenceResolution(coref, sent1)
            except IndexError:
                sent2 = None

        tree = ParentedTree.fromstring(sent2)
        # tree.draw()  # show display
        Positions = getListPos(tree)
        Positions_depths = getOrderPos(Positions)
        Positions_leaves = getLeafPos(tree)
        # locate all VP's in the sentence.
        posOfVP = findPosOfSpecificLabel(tree, "VP", Positions_depths, Positions_leaves)

        ####################
        ######  VP  ########
        ####################
        if posOfVP != None:
            for z in posOfVP:   # iterate over VP's
                Triple = []
                Verb = "Unknown"
                NextStep, found_relation = True, False

                PosInTree = PositionInTree(z, Positions_depths)
                child = findChildNodes(PosInTree, z, Positions_depths)
                for x in child:
                    pos_tag = checkLabel(tree, PositionInTree(x, Positions_depths))
                    if checkLabel(tree, PositionInTree(x, Positions_depths)) == "VP": # VBD?
                        NextStep = False
                    elif pos_tag in relation_tags:  # choose the Last Verb in VP as the Head verb
                        Verb = x
                        found_relation = True
                        # break? out and stop working with this VP
                # If next step still equals true then there is no VP child of the current VP and
                # we can proceed to the next step.
                if NextStep:
                    if not found_relation:
                        Verb = child[0]
                    Verb = findLeavesFromNode(PositionInTree(Verb, Positions_depths), Positions_leaves)
                    Verb = checkLabelLeaf(tree, Verb)

                    if Verb == "dispatched":
                        dud = 0

                    Subject = "Unknown"    # ###################################################################
                    LeftSibling = findLeftSiblingCurrentLevel(z, Positions_depths)
                    LeftSiblingPos = PositionInTree(LeftSibling, Positions_depths)
                    RunCheck = True
                    try:
                        LeftSiblingLabel = checkLabel(tree, LeftSiblingPos)  # Try left-sibling
                    except:
                        RunCheck = False    # No left-sibling
                    if RunCheck and LeftSiblingLabel == "NP":
                        leaves = findLeavesFromNode(LeftSiblingPos, Positions_leaves)
                        for nod in leaves:
                            Subject = leaves[len(leaves) - 1] # leaves[0]
                            tmp = nod[0:len(leaves[0]) - 1]
                            # tmp = leaves[0][0:len(leaves[0]) - 1]  # dod  22 Sept 15
                            tmp2 = checkLabelLeaf(tree, tmp) # test for , and and other odd labels
                            if isinstance(tmp2, str) and tmp2 != "," and "(" in tmp2 \
                                    and tmp2.label() in concept_tags: # NN, NNP, NNS, ...
                                Subject = checkLabelLeaf(tree, Subject)
                    else:
                        # If left sibling isnt a NP then check parent and its NP, repeat until you find NP.
                        CurrentVP = z  # will change to x or something when I loop through all VP's
                        cont, counter = True, 0
                        while cont == True and counter<10:
                            counter+=1               # get parent of this VP
                            Parent = findParentNode(PositionInTree(CurrentVP, Positions_depths), Positions_depths)

                            # now that we have parent, check its leftsibling
                            ParentLeftSibling = findLeftSiblingCurrentLevel(Parent, Positions_depths)
                            # now check the label of the parents left sibling, if it is an NP then use the above code, if it is node then repeat the process
                            ParentLeftSiblingPOS = PositionInTree(ParentLeftSibling, Positions_depths)
                            RunCheck = True
                            try:
                                ParentLeftSiblingPOSLabel = checkLabel(tree, ParentLeftSiblingPOS)
                            except:
                                RunCheck = False

                            if RunCheck and ParentLeftSiblingPOSLabel == "NP":
                                leaves = findLeavesFromNode(ParentLeftSiblingPOS, Positions_leaves)
                                Subject = leaves[len(leaves) - 1]
                                Subject = checkLabelLeaf(tree, Subject)
                                cont = False
                                break
                            else:
                                CurrentVP = Parent

                    # now that I have the subject and Verb I should combine these together and create a double.
                    for ner_n in sentence_ners:
                        if isinstance(Subject, str) and Subject in ner_n:  # always a 1-word subject?
                            #print(Subject, "->", ner_n, end="  ")
                            Subject = ner_n.replace(" ", "_")
                            break
                    if isinstance(Subject, tuple):
                        Subject = "Unknown"
                    Triple.append(Subject)
                    Triple.append(Verb)

                    # now locate the OBJECT - if there is one.
                    Obj = "Unknown"  # ###########################################################################
                    # reuse some of the code from previous rule to find closest NP on the right of the verb.
                    ListOfNP = findPosOfSpecificLabel(tree, "NP", Positions_depths, Positions_leaves)
                    PosOfVerbTree = Positions.index(Positions_depths[child[0][0]][child[0][1]])
                    index = []
                    if ListOfNP:
                        for x in ListOfNP:
                            index.append(Positions.index(Positions_depths[x[0]][x[1]]))

                    closest = 0
                    currentDif = 100000
                    for y in index:
                        diff = y - PosOfVerbTree
                        if (diff > 0 and diff < currentDif): # nearest NP to the right
                            currentDif = diff
                            closest = y

                    # check if closest has an NP child, if it does work from this node instead
                    loop = True
                    count = 0
                    currentNode = findPosInOrderList(Positions[closest], Positions_depths)

                    while loop and count<10:
                        currentNodePOS = PositionInTree(currentNode,Positions_depths)
                        currentNodeChild = findChildNodes(currentNodePOS, currentNode, Positions_depths)
                        currentNodeChildTreePOS = PositionInTree(currentNodeChild[0], Positions_depths)
    
                        if(currentNodeChildTreePOS in Positions_leaves):
                            loop = False
                            break
                        elif checkLabel(tree, currentNodeChildTreePOS)=="NP":
                            currentNode = currentNodeChild[0]
                        else:
                            leaves = findLeavesFromNode(currentNodePOS, Positions_leaves)
                            Obj = checkLabelLeaf(tree, leaves[len(leaves)-1])
                            if Obj == "." and len(leaves)>=2: # or its not an NN
                                Obj = checkLabelLeaf(tree, leaves[len(leaves) - 2])
                            loop = False
                            break
                    if isinstance(Obj, tuple):
                            Subject = "Unknown"
                    if Obj != ".":
                        for ner_n in sentence_ners:
                            if Obj in ner_n:
                                # print(Obj, "->", ner_n, end="   ")
                                Obj = ner_n.replace(" ", "_")
                                break

                        Triple.append(Obj)
                        sentence_triples.append(Triple)  # end PosOfVP for loop

        ####################
        ######  PP  ########
        ####################
        PosOfPP = findPosOfSpecificLabel(tree, "PP", Positions_depths, Positions_leaves)

        if PosOfPP != None:
            for z in PosOfPP:
                Triple = []
                Preposition = "Unknown"
                NextStep = True

                PosInTree = PositionInTree(z, Positions_depths)
                child = findChildNodes(PosInTree, z, Positions_depths)
                for x in child:
                    if checkLabel(tree, PositionInTree(x, Positions_depths)) == "PP": # PP is parent of PP
                        NextStep = False
                if NextStep:
                    Preposition = child[0]
                    Preposition = findLeavesFromNode(PositionInTree(Preposition, Positions_depths), Positions_leaves)
                    if type(Preposition) == list:
                        if len(Preposition[0]) >1 :           ## ERROR from here, need an else?
                            Preposition = [Preposition[0]]
                    Preposition = checkLabelLeaf(tree, Preposition)

                    # find SUBJECT NP on the left
                    PosPPTree = Positions.index(Positions_depths[child[0][0]][child[0][1]])
                    closest = 0
                    currentDif = -1000000

                    """if isinstance(index, int):  # ulgy hack
                        index = [index] """
                    try:
                        index
                    except NameError:  # no Verb
                        index = []

                    for y in index:   # position of Prp$
                        diff = y - PosPPTree
                        if (diff < 0 and diff > currentDif): # left NP
                            currentDif = diff
                            closest = y
                    # now that you have closest NP get the children
                    leaves = findLeavesFromNode(Positions[closest], Positions_leaves)
                    # add the right most leaf to the triple
                    subject = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                    for ner_n in sentence_ners:
                        if subject in ner_n:  # always a 1-word subject?
                            # print(subject, "->", ner_n, end="  ")
                            subject = ner_n.replace(" ", "_")
                            break
                    Triple.append(subject)
                    Triple.append(Preposition)

                    # get OBJECT NP on the right
                    closest = 0
                    currentDif = 100000
                    for y in index:
                        diff = y - PosPPTree
                        if (diff > 0 and diff < currentDif): # right NP
                            currentDif = diff
                            closest = y

                    # check if closet has an NP child, if it does work from child
                    leafLabel, subject = "UNKNOWN", "UNKNOWN"
                    loop = True
                    count = 0

                    if closest>= len(Positions):
                        closest=(len(Positions)-1) ## No NP
                    currentNode = findPosInOrderList(Positions[closest], Positions_depths)

                    while (currentNode != None) and (loop and count<10):   # Why 10? 10 attempts?
                        # ClosestPosInOrderList = findPosInOrderList(Positions[closest],Positions_depths)
                        # childOfClosest = findChildNodes(Positions[closest], ClosestPosInOrderList, Positions_depths)
                        # childOfClosestTreePOS = PositionInTree(childOfClosest[0], Positions_depths)
                        currentNodePOS = PositionInTree(currentNode, Positions_depths)
                        currentNodeChild = findChildNodes(currentNodePOS, currentNode, Positions_depths)
                        if currentNodeChild == []:
                            pass
                        else: 
                            currentNodeChildTreePOS = PositionInTree(currentNodeChild[0], Positions_depths)
                            if (currentNodeChildTreePOS in Positions_leaves):
                                loop = False
                                break
                            elif checkLabel(tree, currentNodeChildTreePOS) == "NP":
                                currentNode = currentNodeChild[0]
                            else:
                                leaves = findLeavesFromNode(currentNodePOS, Positions_leaves)
                                subject = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                                loop = False
                                break
                        count+=1
                    for ner_n in sentence_ners:
                        if subject in ner_n:  # always a 1-word subject?
                            # print(subject, "->", ner_n, end=" ")
                            subject = ner_n.replace(" ", "_")
                            break
                    Triple.append(subject)
                    sentence_triplesPP.append(Triple)

        # *********************************
        # *        POST PROCESSING        *
        # *********************************

        # TODO post processing
        dud=0
        sentence_triples_copy = sentence_triples.copy()
        sentence_triplesPP_copy = sentence_triplesPP.copy()
        for x in sentence_triples_copy:
            if x[0] in illegal_concept_nodes or x[2] in illegal_concept_nodes:
                sentence_triples.remove(x)
            elif x[0] == 'Unknown' or x[2] == 'Unknown':
                sentence_triples.remove(x)
            elif x[0] == ',' or x[2] == ',':
                sentence_triples.remove(x)
            #elif (not re.match(r'^\w+$', x[0]) or not re.match(r'^\w+$', x[1]) or not re.match(r'^\w+$', x[2])):
            #    sentence_triples.remove(x)
            #    print(" DEL ", x, end=" ")

        for x in sentence_triplesPP_copy:  # remove invalid triples
            if x[0] in illegal_concept_nodes or x[2] in illegal_concept_nodes:
                sentence_triplesPP.remove(x)
            elif x[0] == 'Unknown' or x[2] == 'Unknown':
                sentence_triplesPP.remove(x)
            elif x[0] == ',' or x[2] == ',':
                sentence_triplesPP.remove(x)
            #elif (not re.match(r'^\w+$', x[0]) or not re.match(r'^\w+$', x[1]) or not re.match(r'^\w+$', x[2])):
            #    sentence_triplesPP.remove(x)
            #    print(" DEL ", x, end=" ")
        dud = 0
        for vb_triple in sentence_triples:  ## Phrasal verb composition
            for prp_triple in sentence_triplesPP:  # X vb Y followed by  X prp Y in same sentence
                if (prp_triple[0]== vb_triple[0] and prp_triple[2]== vb_triple[2] and
                    (vb_triple[1] +" " +prp_triple[1]) in sents[i]): # sequence bv prp in the text
                    if prp_triple in sentence_triplesPP:       #already removed?
                        sentence_triplesPP.remove(prp_triple)
                    if vb_triple in sentence_triples:
                        sentence_triples.remove(vb_triple)
                    sentence_triples.append([prp_triple[0], vb_triple[1]+"_"+prp_triple[1], prp_triple[2]])
                    #print(vb_triple[1]+"_"+prp_triple[1], end="")

        print("\n",list_of_sentences[i])
        print("VP predicates: ", sentence_triples)
        print("PP predicates: ", sentence_triplesPP, end="  ")
        print()
        #tree.draw()  # show display parse tree
        
        document_triples.append(sentence_triples)
        document_triples.append(sentence_triplesPP)
    return


def process_sentence_DEPRECATED(text):  # full document
    global sentence_triples
    global sentence_triplesPP
    global set_of_raw_concepts
    list_of_sentences = sent_tokenize(text)
    sents = sent_tokenize(text)
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize, ssplit, parse, ner, dcoref',
        'outputFormat': 'json'})
    if output == "CoreNLP request timed out. Your document may be too long.":
        print("** Timeout of the Stanford Parser")
        list_of_sentences = []  # No parsed output to process
    elif isinstance(output, dict):
        try:
            coref = output['corefs']  ## OCCASIONALY DODGY RESULTS - ??? finds no corefs????
        except IndexError:
            coref = None
    elif type(output) is str:  # or type(output) is unicode:
        output = json.loads(output, encoding='utf-8', strict=False)
        coref = output['corefs']
    else:
        print("** Stanford Parser Error - type:", type(output), end="")

    for i in range(len(list_of_sentences)):
        sentence_triples = []
        sentence_triplesPP = []

        try:
            sent1 = output['sentences'][i]['parse']
        except IndexError:
            sent1 = None
        if sent1 is not None:
            try:
                sent2 = CoreferenceResolution(coref, sent1)
            except IndexError:
                sent2 = None

        tree = ParentedTree.fromstring(sent2)
        # tree.draw()  # show display

        Positions = getListPos(tree)

        Positions_depths = getOrderPos(Positions)
        Positions_leaves = getLeafPos(tree)
        # find the children of S
        # TODO implement new set of rule
        # locate all VP's in the sentence.
        posOfVP = findPosOfSpecificLabel(tree, "VP", Positions_depths, Positions_leaves) # Position of VP

        ####################
        ######  VP  ########
        ####################
        if posOfVP == None:
                print("No VP found.")
        else:
            for z in posOfVP:  # iterative over VP's
                Triple = []
                NextStep = True

                PosInTree = PositionInTree(z, Positions_depths)
                child = findChildNodes(PosInTree, z, Positions_depths)
                for x in child:
                    if checkLabel(tree, PositionInTree(x, Positions_depths)) == "VP":
                        NextStep = False  # break out and stop working with this VP
                ###########################
                # If no VP child of the current VP then we can proceede to the next step.
                if NextStep:
                    VerbTree = child[0]
                    Verb = child[0]
                    Verb = findLeavesFromNode(PositionInTree(Verb, Positions_depths), Positions_leaves)
                    Verb = checkLabelLeaf(tree, Verb)
                    Subject = "Unknown"

                    LeftSibling = findLeftSiblingCurrentLevel(z, Positions_depths)
                    LeftSiblingPos = PositionInTree(LeftSibling, Positions_depths)
                    # print(checkLabel(tree, LeftSiblingPos))
                    RunCheck = True
                    try:
                        LeftSiblingLabel = checkLabel(tree, LeftSiblingPos)
                    except:
                        RunCheck = False  # No left-sibling

                    if RunCheck and LeftSiblingLabel == "NP":
                        leaves = findLeavesFromNode(LeftSiblingPos, Positions_leaves)
                        Subject = leaves[len(leaves) - 1]
                        Subject = checkLabelLeaf(tree, Subject)

                    else:  # If left sibling isnt a NP then check parent and its NP, repeat until you find NP.
                        CurrentVP = z  # this will change later to x or something when I loop over all the VP's
                        cont = True
                        counter = 0  # why?

                        while cont == True and counter < 10:
                            counter += 1
                            # get parent of this VP
                            Parent = findParentNode(PositionInTree(CurrentVP, Positions_depths), Positions_depths)
                            # print(CurrentVP)
                            # Parent is the parent node, we have parent  check its leftsibling
                            ParentLeftSibling = findLeftSiblingCurrentLevel(Parent, Positions_depths)
                            # print(ParentLeftSibling)
                            # now check the label of the parents left sibling, if it is an NP then use the above code, if it is node then repeat the process
                            ParentLeftSiblingPOS = PositionInTree(ParentLeftSibling, Positions_depths)
                            RunCheck = True
                            try:
                                ParentLeftSiblingPOSLabel = checkLabel(tree, ParentLeftSiblingPOS)
                            except:
                                RunCheck = False

                            if RunCheck and ParentLeftSiblingPOSLabel == "NP": # trueNP2
                                leaves = findLeavesFromNode(ParentLeftSiblingPOS, Positions_leaves)
                                Subject = leaves[len(leaves) - 1]
                                Subject = checkLabelLeaf(tree, Subject)
                                cont = False
                                break
                            else:
                                CurrentVP = Parent

                    # now that I have the subject and Verb I should combine these together and create a double.
                    if Subject.count("_") >= 2:
                        # print(Subject,"->", end="")
                        Subject = trim_concept_chain(Subject)
                        # print(Subject,"   ", end="")
                    Triple.append(Subject)

                    Triple.append(Verb)  # Partial Triple

                    # now locate the OBJECT - if there is one.
                    Obj = "Unknown-Obj "
                    # reuse some of the code from previous rule to find closest NP on the right of the verb.
                    ListOfNP = findPosOfSpecificLabel(tree, "NP", Positions_depths, Positions_leaves)
                    PosOfVerbTree = Positions.index(Positions_depths[child[0][0]][child[0][1]])

                    index = []
                    if ListOfNP:  # dod
                        for x in ListOfNP:
                            index.append(Positions.index(Positions_depths[x[0]][x[1]]))

                    closest = 0
                    currentDif = 100000
                    for y in index:
                        diff = y - PosOfVerbTree
                        if (diff > 0 and diff < currentDif):
                            currentDif = diff
                            closest = y

                    # check if closest has an NP child, if it does work from this node instead
                    loop = True
                    count = 0
                    currentNode = findPosInOrderList(Positions[closest], Positions_depths)
                    while loop and count < 10:
                        currentNodePOS = PositionInTree(currentNode, Positions_depths)
                        currentNodeChild = findChildNodes(currentNodePOS, currentNode, Positions_depths)
                        currentNodeChildTreePOS = PositionInTree(currentNodeChild[0], Positions_depths)

                        if (currentNodeChildTreePOS in Positions_leaves):
                            loop = False
                            break
                        elif checkLabel(tree, currentNodeChildTreePOS) == "NP":
                            currentNode = currentNodeChild[0]
                        else:
                            leaves = findLeavesFromNode(currentNodePOS, Positions_leaves)
                            Obj = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                            loop = False
                            break

                    if Obj != ".":
                        #if Obj.count("_") >= 2:  # trim coreference Phrases
                        #    Obj = trim_concept_chain(Obj)
                        Triple.append(Obj)
                        # print(" TRIPLE: ", Triple, end="")
                        sentence_triples.append(Triple)  # end PosOfVP for loop

        ####################
        ######  PP  ########
        ####################

        PosOfPP = findPosOfSpecificLabel(tree, "PP", Positions_depths, Positions_leaves)
        # global sentence_triplesPP
        # sentence_triplesPP = []
        if PosOfPP != None:
            for z in PosOfPP:
                Triple = []
                Preposition = ""
                NextStep = True

                PosInTree = PositionInTree(z, Positions_depths)
                child = findChildNodes(PosInTree, z, Positions_depths)
                for x in child:
                    if checkLabel(tree, PositionInTree(x, Positions_depths)) == "PP":
                        NextStep = False  # CheckLabel is True"
                    # CheckLabel is False

                if NextStep:
                    Preposition = child[0]
                    Preposition = findLeavesFromNode(PositionInTree(Preposition, Positions_depths), Positions_leaves)
                    # Preposition index:", Preposition)

                    if type(Preposition) == list:
                        if len(Preposition[0]) > 1:  ## ERROR from here
                            Preposition = [Preposition[0]]
                    # tree[Preposition]", tree[Preposition])

                    Preposition = checkLabelLeaf(tree, Preposition)

                    # find NP on the left
                    PosPPTree = Positions.index(Positions_depths[child[0][0]][child[0][1]])
                    closest = 0
                    currentDif = -1000000

                    """if isinstance(index, int):  # ulgy hack
                        index = [index]
                        #print("is int")   """

                    try:
                        index
                    except NameError:  # no Verb
                        index = []

                    for y in index:  # position of V
                        diff = y - PosPPTree
                        if (diff < 0 and diff > currentDif):
                            currentDif = diff
                            closest = y
                    # now that you have closest NP get the children
                    leaves = findLeavesFromNode(Positions[closest], Positions_leaves)
                    # add the right most leaf to the triple
                    leafLabel = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                    Triple.append(leafLabel)
                    Triple.append(Preposition)

                    # now get NP on the right
                    closest = 0
                    currentDif = 100000
                    for y in index:
                        diff = y - PosPPTree
                        if (diff > 0 and diff < currentDif):
                            currentDif = diff
                            closest = y

                    # check if closet has an NP child, if it does work from child
                    leafLabel = "UNKNOWN"
                    loop = True
                    count = 0

                    if closest >= len(Positions): closest = (len(Positions) - 1)  ## No NP
                    currentNode = findPosInOrderList(Positions[closest], Positions_depths)

                    while (currentNode != None) and (loop and count < 10):  # Why 10? 10 attempts?
                        # check if child is a leaf node first
                        # ClosestPosInOrderList = findPosInOrderList(Positions[closest],Positions_depths)
                        # childOfClosest = findChildNodes(Positions[closest], ClosestPosInOrderList, Positions_depths)
                        # childOfClosestTreePOS = PositionInTree(childOfClosest[0], Positions_depths)
                        currentNodePOS = PositionInTree(currentNode, Positions_depths)
                        currentNodeChild = findChildNodes(currentNodePOS, currentNode, Positions_depths)
                        if currentNodeChild == []:
                            pass
                        else:
                            currentNodeChildTreePOS = PositionInTree(currentNodeChild[0], Positions_depths)
                            if (currentNodeChildTreePOS in Positions_leaves):
                                loop = False
                                break
                            elif checkLabel(tree, currentNodeChildTreePOS) == "NP":
                                currentNode = currentNodeChild[0]
                            else:
                                leaves = findLeavesFromNode(currentNodePOS, Positions_leaves)
                                leafLabel = checkLabelLeaf(tree, leaves[len(leaves) - 1])
                                loop = False
                                break
                        count += 1

                    Triple.append(leafLabel)
                    sentence_triplesPP.append(Triple)

        # *********************************
        # *        POST PROCESSING        *
        # *********************************

        # TODO post processing
        x = 0
        sentence_triples_copy = sentence_triples.copy()
        sentence_triplesPP_copy = sentence_triplesPP.copy()
        for x in sentence_triples_copy:
            if x[0] in illegal_concept_nodes or x[2] in illegal_concept_nodes:
                sentence_triples.remove(x)
            elif x[0] == 'Unknown' or x[2] == 'Unknown':
                sentence_triples.remove(x)
            elif x[0] == ',' or x[2] == ',':
                sentence_triples.remove(x)
            elif (not re.match(r'^\w+$', x[0]) or not re.match(r'^\w+$', x[1]) or not re.match(r'^\w+$', x[2])):
                sentence_triples.remove(x)

        for x in sentence_triplesPP_copy:  # remove invalid triples
            if x[0] in illegal_concept_nodes or x[2] in illegal_concept_nodes:
                sentence_triplesPP.remove(x)
            elif x[0] == 'Unknown' or x[2] == 'Unknown':
                sentence_triplesPP.remove(x)
            elif x[0] == ',' or x[2] == ',':
                sentence_triplesPP.remove(x)
            elif (not re.match(r'^\w+$', x[0]) or not re.match(r'^\w+$', x[1]) or not re.match(r'^\w+$', x[2])):
                sentence_triplesPP.remove(x)
        x = 0
        for vb_triple in sentence_triples:  ## Phrasal verb composition
            for prp_triple in sentence_triplesPP:  # X vb Y followed by  X prp Y in same sentence
                if (prp_triple[0] == vb_triple[0] and prp_triple[2] == vb_triple[2] and
                        (vb_triple[1] + " " + prp_triple[1]) in sents[i]):  # sequence bv prp in the text
                    if prp_triple in sentence_triplesPP:  # already removed?
                        sentence_triplesPP.remove(prp_triple)
                    if vb_triple in sentence_triples:
                        sentence_triples.remove(vb_triple)
                    sentence_triples.append([prp_triple[0], vb_triple[1] + "_" + prp_triple[1], prp_triple[2]])
                    print(vb_triple[1] + "_" + prp_triple[1])

        #print("\nsentence_triples: ", sentence_triples)
        #print("sentence_triplesPP: ", sentence_triplesPP, end="  ")
        #print()
        # tree.draw()  # show display parse tree

        document_triples.append(sentence_triples)
        document_triples.append(sentence_triplesPP)

    return


# *********************************
# *   OUTPUT PREPARATION        *
# *********************************

def generate_output_CSV_file(fileName):
    global document_triples
    testList = BringListDown1D(document_triples)
    heading = [["NOUN", "VERB/PREP", "NOUN"]]
    with open(out_path + fileName+".dcorf.csv", 'w', encoding="utf8") as resultFile:
        write = csv.writer(resultFile, lineterminator='\n')
        write.writerows(heading)
        write.writerows(testList)
    resultFile.close()
    return


def processAllTextFiles():
    global in_path
    global document_triples
    global sentence_number
    global set_of_raw_concepts
    global skip_over_previous_results
    fileList = os.listdir(in_path)
    txt_files = [i for i in fileList if i.endswith('.txt')]
    #txt_files = [i for i in txt_files if i.startswith('TEMP')]
    if 'Cache.txt' in txt_files:
        txt_files.remove('Cache.txt')  # System file for Text2Predic8
    csv_files = [i for i in fileList if i.endswith('.csv')]
    for fileName in txt_files:
        set_of_raw_concepts = set()
        sentence_number = 0
        print("\n####################################################")
        print("FILE ", in_path, "&&", fileName)
        if skip_over_previous_results and path.isfile(in_path + fileName + ".dcorf.csv"):
            print(" £skippy ", end="")
            continue
        global data
        data = []
        document_triples = []
        try:
            file = open(in_path+fileName, "r", encoding="utf8", errors='replace')
        except Exception as err:
             print("Erro {}".format(err))
        full_document = file.read()
        full_document_list = full_document.split()
        text_chunk_size = 500
        text_chunk_start = 0
        text_chunk_end = text_chunk_size
        while text_chunk_start < len(full_document_list):
            z = full_document_list[text_chunk_start:text_chunk_end]
            documentSegment = " ".join(z)
            processDocument(documentSegment)
            text_chunk_start += text_chunk_size -2
            text_chunk_end = text_chunk_end + text_chunk_size
            #if text_chunk_start >=4500:
            #    print("End Of Document Truncated", end="")
            if text_chunk_end > len(full_document_list):
                text_chunk_end = min(len(full_document_list),len(full_document_list))
            else:
                if text_chunk_start < len(full_document_list):
                    while (text_chunk_end >= text_chunk_start + (text_chunk_size - 100)) and \
                        ("." not in full_document_list[text_chunk_end] or \
                        "?" not in full_document_list[text_chunk_end] or \
                        ":" not in full_document_list[text_chunk_end]):  # split chunks @ full-stop, where reasonable
                        text_chunk_end -= 1
        generate_output_CSV_file(fileName) # uses documentSegment, documentTriples
        set_of_unique_concepts = set()
        for tripl in document_triples:
            for a,b,c in tripl:
                set_of_unique_concepts.add(a)
                set_of_unique_concepts.add(c)
        print("#CONCEPTS = ", len(set_of_unique_concepts), " #RELATIONS = ", len(document_triples), " in ", fileName)
    return


def add_line_to_output_CSV_file(fileName):
    global document_triples
    testList = BringListDown1D(document_triples)
    heading = [["NOUN", "VERB/PREP", "NOUN"]]
    with open(out_path + fileName+".dcorf.csv", 'w', encoding="utf8") as resultFile:
        write = csv.writer(resultFile, lineterminator='\n')
        write.writerows(heading)
        write.writerows(testList)
    resultFile.close()
    return


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def count_words(in_string):
    return len(re.findall(r'\w+', in_string))

def count_content_overlap(string_a, string_b):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))
    word_tokens_a = word_tokenize(string_a)
    word_tokens_b = word_tokenize(string_b)
    filtered_sentence_a = [w for w in word_tokens_a if not w.lower() in stop_words]
    filtered_sentence_b = [w for w in word_tokens_b if not w.lower() in stop_words]
    return len(intersection(filtered_sentence_a, filtered_sentence_b))
""""   filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w) """


print("1) Set your java path in line 19.")
print("2) Change the base_path variable if it is Not the current working directory.", base_path)
print("Type   processAllTextFiles()   to generate graphs from .txt files from ", in_path)
print()

# processDocument("John Smyth drove his new car but he crashed it. ")
# processDocument("John Smyth drove his new car but he crashed it. It was a bad day for Jane Doe. \
#                Tom Sullivan was eating in Sunbury on Thames that day.")

#processDocument("POLITICAL LEADERS PAST and present have paid tribute to David Trimble’s contribution to peace \
# in Northern Ireland following his death. The 77-year-old ex-leader of the Ulster Unionist Party \
# was one of the principal architects of the Good Friday Agreement that ended decades of conflict in \
# the region. Trimble, who jointly won the Nobel Peace Prize along with late SDLP leader John Hume, \
# died yesterday following an illness.")

# processDocument("The strongest rain ever recorded in India shut down the financial hub of Mumbai, snapped \
# communication lines, closed airports and forced thousands of people to sleep in their offices or walk home \
# during the night, officials said today.")

processDocument("John drove his new car but he crashed it.")
processDocument("The fourth Wells account moving to another agency is the packaged paper-products division of Georgia-Pacific Corp., which arrived at Wells only last fall. Like Hertz and the History Channel, it is also leaving for an Omnicom-owned agency, the BBDO South unit of BBDO Worldwide. BBDO South in Atlanta, which handles corporate advertising for Georgia-Pacific, will assume additional duties for brands like Angel Soft toilet tissue and Sparkle paper towels, said Ken Haldin, a spokesman for Georgia-Pacific in Atlanta.")
processAllTextFiles()

#import nltk


