from buffer import *

from CG1 import *
from chunks import *
import numpy as np
from time import time

def test_abstraction():
    cg = CG1.CG1()
    c1 = Chunk({(0,2,1,1),(0,2,2,1)},count = 20)
    c2 = Chunk({(0,2,1,1),(0,3,2,1)},count = 20)
    c3 = Chunk({(0,3,2,1),(0,2,2,1)}, count = 20)
    cg.concrete_chunks = [c1,c2,c3]
    c1.adjacency = {'1': {'1': 10, '0':1, '2':5}, '0': {'1':5, '0':3, '2':2}}
    c2.adjacency = {'1': {'1': 2, '0':1, '2':1}, '0': {'1':2, '0':4, '2':5}}
    c2.adjacency = {'1': {'1': 3, '0': 1, '2': 2}, '3': {'1': 2, '0': 1, '2': 5}}
    cg.visible_chunk_list = [c1.content, c2.content, c3.content]
    cg = abstraction(cg) # or something that cg.abstraction to make it a graph pruning routine in cg.


def test1_recursive_match():
    cg = CG1()
    two = Chunk([(0, 0, 0, 2.0)])
    three = Chunk([(0, 0, 0, 3.0)])
    four = Chunk([(0, 0, 0, 4.0)])
    five = Chunk([(0, 0, 0, 5.0)])
    one = Chunk([(0, 0, 0, 1.0)])

    for item in [one, two, three, four, five]:
        cg.add_chunk(item)

    threeorfour = Variable([three, four], cg)
    oneorfive = Variable([one, five], cg)

    twov1v2 = Chunk([(0, 0, 0, 2)])  # overwirting
    twov1v2.ordered_content = two.ordered_content.copy()

    twov1v2.ordered_content['1'] = threeorfour
    threeorfour.chunks[twov1v2.key] = twov1v2

    twov1v2.ordered_content['2'] = oneorfive
    oneorfive.chunks[twov1v2.key] = twov1v2

    cg.add_chunk(twov1v2)

    seqc = {(0, 0, 0, 2.0), (1, 0, 0, 3.0), (2, 0, 0, 7.0), (3, 0, 0, 3)}

    match, matchingcontent = check_recursive_match(seqc, set(), twov1v2)
    return


def test2_recursive_match():
    cg = CG1()
    two = Chunk([(0, 0, 0, 2.0)])
    three = Chunk([(0, 0, 0, 3.0)])
    four = Chunk([(0, 0, 0, 4.0)])
    five = Chunk([(0, 0, 0, 5.0)])
    one = Chunk([(0, 0, 0, 1.0)])

    for item in [one, two, three, four, five]:
        cg.add_chunk(item)

    threeorfour = Variable([three, four], cg)
    twov1 = Chunk(list(two.ordered_content['0']))
    twov1.ordered_content = two.ordered_content.copy()
    twov1.ordered_content['1'] = threeorfour
    cg.add_chunk(twov1)
    threeorfour.chunks[twov1.key] = twov1

    twov1orfive = Variable([twov1, five], cg)
    onetwov1orfive = Chunk(list(one.ordered_content['0']))
    onetwov1orfive.ordered_content = one.ordered_content.copy()
    onetwov1orfive.ordered_content['1'] = twov1orfive
    cg.add_chunk(onetwov1orfive)
    twov1orfive.chunks[onetwov1orfive.key] = onetwov1orfive
    seqc = {(0, 0, 0, 1.0), (2, 0, 0, 5.0), (1, 0, 0, 9.0)}

    match, matchingcontent = check_recursive_match(seqc, set(), onetwov1orfive)
    print(check_recursive_match(seqc, set(), onetwov1orfive))
    return


def abstraction(cg):
    '''input: a chunking graph
        output: the chunking graph organized in a tree structure'''
    cg.network = {}
    # use relative value position to encode parsing table.
    A = {(1,2,2,1),(1,3,2,4),(1,4,2,3)}
    B = {(1,2,2,1),(1,3,2,2)}
    C = {(0,3,2,1),(1,3,2,2)}

    A & B
    B & C
    dictionary = {(1,2,2,1): [A],}
    for thisck in cg.chunks:  # find the chunk that best optimizes average compression,
        '''Find common content amongst all concrete chunks'''
        # comment: there will be n_chunk choose 2 number of possible intersections.

        # when thisck is a concrete, non-abstract chunk
        # find all unique intersections amongst a pair of concrete chunks
        all_abstractions = []
        proposed_abstractions = set()  # all ways of extracting the abstraction of this chunk

        for ck in cg.concrete_chunks:  # find all intersection between this chunk and all other concrete chunks
            intersection = ck.content & thisck.content
            A = ck.content - intersection
            B = thisck.content - intersection
            # combine intersection, A and B into a variable.
            variable = {A, B}


            intersection = change_ref_frame(intersection)

            entailmentchunks = []
            if intersection not in abstractions and len(intersection) < len(ck.content) and len(intersection) > 0:
                freq = ck.count + thisck.count  # abstraction frequency
                for _ck in cg.concrete_chunks:  # how general is this abstraction?
                    if ck != _ck and thisck != _ck and intersection.issubset(
                            _ck.content):  # comment: can simplify computation by accounting for recent memory
                        freq = freq + _ck.count
                        entailmentchunks.append(_ck)
                abstr_ck = Chunk(intersection, count=freq, entailment=entailmentchunks)
                all_abstractions.append(abstr_ck)
                proposed_abstractions.add(intersection)
        # evaluate the abstraction advantage of all candidate abstractions
        selected_abstraction = rational_abstraction(proposed_abstractions)
        # rearrange graph
    return

def change_ref_frame(intersection):
    # subtract by the biggest number in the content set on the time dimension
    intersection[:,0] = intersection[:,0] - min(intersection[:,0])
    return {tuple(a) for a in intersection}


def rational_abstraction(cg, proposed_abstractions): # or, cg.rational_abstraction
    for abstr_ck in proposed_abstractions:
        # decide on whether this abstract chunk with its discovered intersection within should be integrated into the abstraction graph
        evaluate_abstraction_concrete(cg, abstr_ck)



def evaluate_abstraction_advantage(self, abstr_ck):

    # """How does one abstraction help with representation improvement"""
    # abstraction_chunk = chunks.Chunk(abstraction, count = freq) # create an abstraction
    # for chunks in self.chunks:
    #
    # # for one representation
    # info = 0
    # for a in cg.abstract_chunks:
    #     info = info + evaluate_info(a)
    pass



def evaluate_info(chunk):
    if chunk.entailment == []:# concrete chunks
        return -np.log(chunk.p)
    else: # abstraction
        info = 0
        for ck in chunk.entailment:
            info = info + evaluate_info(ck)
        return info


# each chunk stores p(e|a), the probability of instantiation given an abstraction chunks

def generate_abstraction_tree(cg):
    # returns, vertex and edges, in addition to vertex locations of an abstract graph, and the binary encoding of
    # abstract chunks
    pass

def change_representation(cg, abstr_chunk, entailment,abstraction):
    # find the immediate entailment and immediate abstraction for an abstract chunk
    cgnew = cg.copy()
    cgnew.chunks.append(abstraction)
    cgnew.abstract_chunks.append(abstraction)
    for chunk in entailment:
        abstraction.entailment.append(chunk)
        chunk.count = chunk.count - abstraction.count
        chunk.abstraction.append(abstraction)

    # given an abstraction, point the abstraction to all its entailed abstract and non-abstract chunks
    return cgnew


# parsing with abstraction structure embedded in the hierarchy
def parsing():

    pass

def variable_finding(self, cat):
    v = 3 # threshold of volume of intersection
    app_t = 3# applicability threshold
    '''cat: new chunk which just entered into the system
    find the intersection of cat with the pre-existing chunks '''
    # (content of intersection, their associated chunks) ranked by the applicability threshold
    # alternatively, the most applicable intersection:
    max_intersect = None
    max_intersect_count = 0
    max_intersect_chunks = [] # chunks that needs to be merged
    for ck in self.chunks:
        intersect = ck.content.intersection(cat.content)
        intersect_chunks = []
        c = 0  # how often this intersection is applicable across chunks
        if len(intersect) != len(cat.content) and len(intersect) > v:# not the same chunk
            # look for overlap between this intersection and other chunks:
            for ck_ in self.chunks:# how applicable is this intersection, to other previously learned chunks
                if ck_.content.intersection(intersect) == len(intersect):
                    c = c + 1
                    intersect_chunks.append(ck_)
        if c > max_intersect_count and c >= app_t:
            # atm. select intersect with the max intersect count
            # TODO: can be ranked based on merging gain
            max_intersect_count = c
            max_intersect_chunks = intersect_chunks
            max_intersect = intersect
    if max_intersect!=None: # reorganize chunk list to integrate with variables
        self.merge_chunks(max_intersect, max_intersect_chunks, max_intersect_count)
    return

def merge_chunks(self, max_intersect, max_intersect_chunks, max_intersect_count):
    # create a new chunk with intergrated variables.
    for ck in max_intersect_chunks:
        ck.content = ck.content - max_intersect
    var = Variable(max_intersect_chunks, totalcount=max_intersect_count)
    self.set_variable_adjacency(var, max_intersect_chunks)

    chk = None # find if intersection chunk exists in the past
    for ck in self.chunks:
        if ck.content.intersection(max_intersect) == len(ck.content):
            chk = ck
    if chk == None: #TODO: add new chunk here
        chk = Chunk(max_intersect, count=max_intersect_count)
    else:
        chk.count = max_intersect_count

    # TODO: add new variable chunk here.
    chk_var = Chunk([chk, var])# an agglomeration of chunk with variable is created


    return




def probabilistic_parsing(cg, seq, chunk_record, t):
    '''parse and store chunks in the chunkrecord '''
    def check_obs(s):  # there are observations at the current time point
        if s == []: return True # in the case of empty sequence
        else: return s[0][0] != 0 # nothing happens at the current time point
        # identify biggest chunks that finish at the earliest time point.
    seq_explained = False
    seqc = seq.copy()
    explainchunk = []
    no_observation = check_obs(seqc)# no observation in this particular time slice
    if no_observation:
        current_chunks_idx = []
        dt = int(1.0)
    else:
        while check_seq_explained(seqc) == False:
            # find explanations for the upcoming sequence
            # record termination time of each biggest chunk used for explanation
            # TODO: identify and update chunk, reduce identified part of the sequence
            paths = []
            contents = []
            for ck in cg.ancestors:
                candidate_path = []
                candidate_content = set()
                if ck.content.issubset(seq):
                    while ck.entailment!=[]:
                        candidate_path.append(ck)
                        candidate_content = candidate_content + ck.content

                        for chunk in ck.entailment:
                            if chunk.content.issubset(seq):
                                ck = chunk

                paths.append(candidate_path)
                contents.append(candidate_content)


                    # search until the bottom of the tree
            cg, seqc, chunk_record, explainchunk = identify_one_chunk(cg, seqc, explainchunk, chunk_record, t)


            def check_chunk_in_seq(chunk, seq):
                content = chunk.content
                if content.issubset(seq):
                    return chunk.volume, chunk.T
                else:
                    return 0, 0


        # decide which are current chunks so that time is appropriately updated
        explainchunk.sort(key=lambda tup: tup[1])# sort according to finishing time
        if len(seqc)>=1:
            dt = min(min(explainchunk, key=lambda tup: tup[1])[1], seqc[0][0])
        else:
            dt = min(explainchunk, key=lambda tup: tup[1])[1]
        current_chunks_idx = [item[0] for item in explainchunk if item[1] == dt] # move at the pace of the smallest identified chunk
        seq = seqc
    return current_chunks_idx, cg, dt, seq, chunk_record