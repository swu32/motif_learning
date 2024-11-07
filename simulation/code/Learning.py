import numpy as np
import pandas as pd
from numpy.core._multiarray_umath import ndarray
from scipy import stats
from scipy.stats import chisquare
from math import log2
from chunks import *
from buffer import *
# from test_Learning import *


def hcm_rational_v1(arayseq, cg):
    seql, H, W = arayseq.shape
    cg.update_hw(H, W)
    seq, seql = convert_sequence(arayseq)  # loaded with the 0th observation
    seq = set(seq)
    seq = set(
        [(0, 0, 0, 1), (0, 0, 1, 3), (0, 2, 2, 2), (1, 0, 0, 1), (1, 0, 1, 3), (1, 2, 2, 2), (2, 0, 0, 1), (2, 0, 1, 3),
         (2, 2, 2, 2)])
    thischunk = Chunk([(0, 0, 0, 1), (0, 0, 1, 3), (0, 2, 2, 2)], H=1, W=1)
    cont = list(thischunk.content)
    for obs in seq:
        inseq = False
        matching_elements = set()

        if obs[3] == cont[0][3]:
            t, x, y = obs[0] - cont[0][0], obs[1] - cont[0][1], obs[2] - cont[0][2]
            inseq = True
            matching_elements.add(obs)

            i = 1
            while i < len(cont) and inseq:
                point = cont[i]
                translated_chunk_point = (point[0] + t, point[1] + x, point[2] + y, point[3])
                if translated_chunk_point not in seq:
                    inseq = False
                else:
                    matching_elements.add(translated_chunk_point)
                i = i + 1

        if inseq:
            print('t = ', t)
            # seq.difference_update(matching_elements)
    return


def parse_sequence(cg, seq, arayseq, seql=1000, candidate_set= set()):
    # TODO: check whether the time span of the maximal matching chunk is dynamic/adapts with identified variables
    ''' candidate_set: set of chunks that are parsing candidates'''
    t = 0
    Buffer = buffer(
        t, seq, seql, arayseq.shape[0]
    )  # seql: length of the current parsing buffer
    seq = seq  # reset the sequence
    cg.empty_counts()  # empty out all the counts before the next sequence parse
    if len(candidate_set) < 1:
        candidate_set = set(cg.chunks.values())
    maxchunksize = cg.getmaxchunksize()
    chunk_record = {}  # chunk ending time, and which chunk is it.
    seq_over = False
    i = 0  # seqeunce items, for debugging proposes
    while seq_over == False:
        # identify latest ending chunks
        candidate_set = set(cg.chunks.values())
        current_chunks, cg, dt, seq, chunk_record = identify_latest_chunks(
            cg, seq, chunk_record, Buffer.t, candidate_set
        )  # chunks
        seq = Buffer.refactor(seq, dt)
        i = i + 1

        cg = learning_and_update(
            current_chunks, chunk_record, cg, Buffer.t, threshold_chunk=False)
        maxchunksize = cg.getmaxchunksize()
        Buffer.reloadsize = maxchunksize + 1
        Buffer.checkreload(arayseq)
        seq_over = Buffer.checkseqover()
    print('in total, there are ', i, 'number of items being parsed')
    return cg, chunk_record


epsilon = 0.0000001
def parsing(cg, seq, arayseq, nit, seql=1000, ABS = True, MAXit = 20):
    '''updates the transition and marginals via parsing the sequence'''
    cg.empty_counts()  # always empty the number of counts for a chunking graph before the next parse
    cg, chunkrecord = parse_sequence(cg, seq, arayseq, seql=seql)
    #cg.rep_cleaning() # disable for now, creates problems with deletion

    if nit >= MAXit: return cg
    # if nit >=3:
    #     if cg.learning_data[-1][5] == cg.learning_data[-2][5]: # No new chunks are being created
    #         return cg

    chunklearnable = True # learn chunks when there are no abstraction
    if ABS:# integrate abstraction
        if cg.prev is None:
            chunklearnable = True
        else:
            if cg.prev == 'chunking': # alternate between chunking and abstraction
                chunklearnable = False  #
            elif cg.prev == 'abstraction':
                chunklearnable = True

    # chunklearnable = cg.learning_data[-1][3] > cg.learning_data[-2][3] + epsilon # trajectory of improvement
    # chunklearnable = cg.independence_test() == False

    if chunklearnable:
        cg = evaluate_representation(cg, chunkrecord, seql) # evaluate only during chunking iterations
        cg.prev = 'chunking'
        return chunking(cg, seq, arayseq, nit, ABS = ABS, seql=seql, MAXit=MAXit)
    else:  # no chunks to learn, learn abstraction instead
        cg.prev = 'abstraction'
        return abstraction(cg, chunkrecord, seq, arayseq, nit, ABS = ABS, seql=seql, MAXit=MAXit)


def chunking(cg, seq, arayseq, nit, ABS = True, seql = 1000, MAXit = 20):
    nit = nit + 1
    cg, cp = rational_learning(cg, n_update=1000)  # rationally learn until loss function do not converge
    return parsing(cg, seq, arayseq, nit, ABS=ABS, seql = seql, MAXit=MAXit)  # parse again and update chunks again


def abstraction(cg, chunkrecord, seq, arayseq, nit, ABS = True, seql = 1000, MAXit = 20):
    if ABS:
        cg.abstraction_learning()
    #cg = evaluate_representation(cg, chunkrecord)
    return parsing(cg, seq, arayseq, nit, ABS=ABS, seql = seql, MAXit=MAXit)  # parse again and update chunks again

def curriculum2():
    print('parsing')
    '''updates the transition and marginals via parsing the sequence'''
    while nit <=maxiter:
        print('parsing')
        '''updates the transition and marginals via parsing the sequence'''
        cg.empty_counts()  # always empty the number of counts for a chunking graph before the next parse
        cg, chunkrecord = parse_sequence(cg, seq, arayseq, seql=seql)
        cg.rep_cleaning()
        cg = evaluate_representation(cg, chunkrecord, seql)
        nit = nit + 1

        cg.empty_counts()  # always empty the number of counts for a chunking graph before the next parse
        cg, chunkrecord = parse_sequence(cg, seq, arayseq, seql=seql)
        cg.rep_cleaning()
        cg = evaluate_representation(cg, chunkrecord, seql)

        cg, cp = rational_learning(cg, n_update=10)  # rationally learn until loss function do not converge
        nit = nit + 1
        cg.abstraction_learning()
        cg = evaluate_representation(cg, chunkrecord)

    return cg




def hcm_markov_control(arayseq, cg, ABS = True, MAXit = 20):
    """chunking and abstraction learning structured in the shape of a finite state machine"""
    seql, H, W = arayseq.shape
    cg.update_hw(H, W)# update initial parameters
    seq, seql = convert_sequence(arayseq[:, :, :])# loaded with the 0th observation
    cg = parsing(cg, seq, arayseq, 0, seql, ABS=ABS, MAXit= MAXit)# enter into the state control loop
    return cg



def hcm_depth_parsing(arayseq, cg):
    """chunking and abstraction learning structured in the shape of a finite state machine"""
    seql, H, W = arayseq.shape
    cg.update_hw(H, W)  # update initial parameters
    seq, seql = convert_sequence(arayseq[:, :, :])  # loaded with the 0th observation

    # first, parse the sequence in the ground level and learn chunks until no more chunks can be learned anymore,
    # or up to a certain iteration level/complexity limitation, whatever comes first
    cg, chunkrecord = parsing_and_chunking_at_concrete_level(cg, seq, arayseq, seql=1000, maxit = 4)

    # a for loop until the final iteration

    # second, parse the sequence in the abstract level and learn chunks until no more chunks can be learned anymore,
    cg.abstraction_learning()# learn variables and variable chunks from transition matrix

    cg.empty_counts()  # always empty the number of counts for a chunking graph before the next parse
    cg, chunkrecord = meta_parse_sequence(cg, chunkrecord)

    nit = 0
    chunklearnable = True
    maxit = 6
    while chunklearnable: # learn x new chunks at a time,
        cg, cp = rational_learning(cg, n_update=10, complexity_limit=-np.log(0.05))  # rationally learn until loss function do not converge
        cg, chunkrecord = meta_parse_sequence(cg, chunkrecord)
        nit = nit + 1
        if cg.independence_test() == True or nit >=maxit: # or add complexity limit
            chunklearnable = False

    return cg





def parsing_and_chunking_at_concrete_level(cg, seq, arayseq, seql=1000, maxit = 1):
    '''parse and learn chunks in sequences at a concrete level'''
    cg.empty_counts()  # always empty the number of counts for a chunking graph before the next parse
    cg, chunkrecord = parse_sequence(cg, seq, arayseq, seql=seql)
    nit = 0

    chunklearnable = True
    while chunklearnable: # learn x new chunks at a time,
        cg, cp = rational_learning(cg, n_update=10, complexity_limit=-np.log(0.05))  # rationally learn until loss function do not converge
        cg, chunkrecord = parse_sequence(cg, seq, arayseq, seql=seql)
        nit = nit + 1
        if cg.independence_test() == True or nit >=maxit: # or add complexity limit
            chunklearnable = False

    return cg, chunkrecord




def hcm_rational(arayseq, cg):
    """ returns chunking graph based on rational chunk learning
            Parameters:
                    arayseq(ndarray): Observational Sequences
                    cg (CG1): Chunking Graph
            Returns:
                    cg (CG1): Learned Representation from Data
    """
    learning_data = []
    nU = 5  # number of updates
    seql, H, W = arayseq.shape
    cg.update_hw(H, W)  # update initial parameters
    seq, seql = convert_sequence(arayseq[:, :, :])  # loaded with the 0th observation

    Iter = 0
    maxIter = 20  # up to 20 chunk updates
    while cg.independence_test() == False and Iter <= maxIter:
        print("============ empty out sequence ========== ")
        cg.empty_counts()
        cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.chunks.items()))
        pl, rc, ev, sc, re = evaluate_representation(cg, chunkrecord, seql)
        cg.learning_data.append([seql, pl, rc, ev, sc, re])
        cg, cp = rational_learning(cg, n_update=nU)  # rationally learn until loss function do not converge
        Iter = Iter + 1

    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.ancestors))
    pl, rc, ev = evaluate_representation(cg, chunkrecord, seql)
    learning_data.append([seql, pl, rc, ev, sc, re])

    cg.abstraction_learning()
    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.ancestors))
    pl, rc, ev = evaluate_representation(cg, chunkrecord, seql)
    learning_data.append([seql, pl, rc, ev])

    # cg.empty_counts()
    cg, chunkrecord = meta_parse_sequence(chunkrecord, cg)
    cg, cp = rational_learning(cg, n_update=10)  # rationally learn until loss function do not converge
    test = Test()
    print(test.sample_concrete_content(list(cg.chunks.keys())[30], cg))
    cg.abstraction_learning()

    pl, rc, ev = evaluate_representation(cg, chunkrecord, seql)
    learning_data.append([seql, pl, rc, ev])

    cg.empty_counts()
    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.ancestors))
    cg, chunkrecord = meta_parse_sequence(chunkrecord, cg)
    cg, cp = rational_learning(cg)  # learn chunks with variables

    cg.abstraction_learning()
    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.chunks))
    cg, chunkrecord = meta_parse_sequence(chunkrecord, cg)

    pl, rc, ev = evaluate_representation(cg, chunkrecord, seql)
    learning_data.append([seql, pl, rc, ev])

    return cg


def hcm_rational_curriculum_1(arayseq, cg):
    """ rational chunking with the implementation of abstraction and meta-parsing
    """
    learning_data = [[0, 0, 0, 0, 0, 0]]
    nU = 5  # number of updates
    sl = 0  # sequence length
    seql, H, W = arayseq.shape

    cg.update_hw(H, W)
    seq, seql = convert_sequence(arayseq[:, :, :])  # loaded with the 0th observation
    print("============ empty out sequence ========== ")

    cg.empty_counts()
    cg, chunkrecord = parse_sequence(cg, arayseq.shape[0], seq, seql, candidate_set=set(cg.chunks.items()))
    cg = evaluate_representation(cg, chunkrecord, seql)
    ev = cg.learning_data[-1][3]
    while ev > learning_data[-1][3]:  # learn until the explanatory volume stops increasing
        cg, cp = rational_learning(cg, n_update=nU)  # rationally learn until loss function do not converge
        cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.chunks.items()))
        pl, rc, ev, sc, re = evaluate_representation(cg, chunkrecord, seql)
        sl = sl + seql

    print('===============================')
    cg.abstraction_learning()
    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.chunks.items()))
    pl, rc, ev, sc, re = evaluate_representation(cg, chunkrecord, seql)
    sl = sl + seql
    learning_data.append([sl, pl, rc, ev, sc, re])

    cg, chunkrecord = meta_parse_sequence(chunkrecord, cg)
    pl, rc, ev, sc, re = evaluate_representation(cg, chunkrecord, seql)
    sl = sl + seql
    learning_data.append([sl, pl, rc, ev, sc, re])

    cg, cp = rational_learning(cg, n_update=nU)  # rationally learn until loss function do not converge
    pl, rc, ev, sc, re = evaluate_representation(cg, chunkrecord, seql)
    sl = sl + seql
    learning_data.append([sl, pl, rc, ev, sc, re])

    cg.abstraction_learning()
    pl, rc, ev, sc, re = evaluate_representation(cg, chunkrecord, seql)
    sl = sl + seql
    learning_data.append([sl, pl, rc, ev, sc, re])

    cg.empty_counts()
    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.ancestors))
    cg, chunkrecord = meta_parse_sequence(chunkrecord, cg)
    cg, cp = rational_learning(cg)  # learn chunks with variables

    cg.abstraction_learning()
    cg, chunkrecord = parse_sequence(cg, arayseq, seq, seql, candidate_set=set(cg.chunks))
    cg, chunkrecord = meta_parse_sequence(chunkrecord, cg)

    pl, rc, ev, sc, re = evaluate_representation(cg, chunkrecord, seql)
    sl = sl + seql
    learning_data.append([sl, pl, rc, ev, sc, re])

    test = Test()
    print(test.sample_concrete_content(list(cg.chunks.keys())[30], cg))
    return cg


def evaluate_representation(cg, chunkrecord, seql=0):
    if seql == 0:  # abstraction:
        seql = cg.learning_data[-1][0]
    pl = cg.calculate_pl()  # parsing length
    rc = cg.calculate_rc()  # representation complexity
    ev = cg.calculate_explanatory_volume(len(chunkrecord), seql)  #
    sc = cg.calculate_sequence_complexity(chunkrecord, cg.chunks | cg.variables)
    re = cg.calculate_representation_entropy(chunkrecord)
    stc = cg.calculate_storage_cost()
    nc = len(cg.chunks)
    nv = len(cg.variables)
    cg.learning_data.append([seql, pl, rc, ev, sc, re, nc, nv, stc])
    return cg


def meta_parse_sequence(cg, chunk_record):
    ''' based on chunk_record, search upward (from specific to concrete) in the representation tree to find consistent variables
    to explain the sequence.
        Input: chunkrecord: the parse
        Can use a while loop to obtain all possible variable representations of the same sequence, the number of chunk
        record would be the depth of the representation graph.
        Each while loop would have a transition probability update, and a set of chunks to learn

        Output: a sequence, parsed via the variables that points to the concrete chunks in the chunkrecord
        '''
    ts = sorted(list(chunk_record.keys())) # chunk ending time stamps
    tl = len(ts)
    abstract_chunk_record = {}# output, identified abstractions
    cg.empty_counts()# clear out all transitions and marginals for the next parses
    i = 0
    while i < tl: # go through chunkrecord
        # identify the biggest explaining chunk for the upcoming sequence
        thisT = ts[i]
        maxchunk, maxchunkcontent, maxl, max_delta_i = identify_biggest_meta_chunk(chunk_record, cg, i, ts)  # identify and pop chunks in sequence
        endidx = i + max_delta_i - 1
        cg, abstract_chunk_record = updatechunkabstract(maxchunk, abstract_chunk_record, cg, endidx, ts, maxl)
        t = ts[endidx]# the time stamp of the last chunk being parsed in the sequence
        # write to abstract chunk record
        cg = learning_and_update([maxchunk.key], abstract_chunk_record, cg, t, threshold_chunk=False)
        # remove chunk from seq
        i = i + max_delta_i  # move to the next parse in chunk record
    return cg, abstract_chunk_record

def identify_biggest_meta_chunk(chunkrecord, cg, i, ts):
    '''Identify the biggest meta chunk to parse the sequence in one-step higher meta level
    :param i: the index of the sequence that ck ends at
    :param chunkrecord: the chunk record of the sequence
    :param cg: the chunk graph
    :return
        max_chunk: the biggest meta chunk that is consistent with a sequence that starts with ck
        maxchunkcontent: the content of the biggest meta chunk, to update the chunk record with
        maxl: the length of the biggest meta chunk, measured in the unit of chunks
        i: the index of the sequence that max_chunk ends at
    the content of the biggest meta chunk, the length of the biggest meta chunk'''

    maxl = 0 # the temporal length of the biggest meta-chunk
    maxchunk = None
    maxchunkcontent = None
    max_delta_i = 0
    thiskey = chunkrecord[ts[i]][0][0] # the key of the chunk that ends at this time stamp
    thischunk = cg.chunks[thiskey] if thiskey in cg.chunks else cg.variables[thiskey]
    currentcandidates = [thischunk] + list(thischunk.abstraction.values())  # start searching from singleton chunks, including the variables
    matching_chunks = set()  # the set of all chunks which start with thischunk that is consistent with the chunkrecord sequence

    # the best explaining chunk corresponding to this seed
    c_stars = None
    c_stars_T = None
    c_stars_delta_i = None
    while len(currentcandidates) > 0:  # search until the bottom of the parsing tree from abstract to concrete
        new_c_stars, new_c_stars_T, new_c_stars_delta_i = check_meta_match(chunkrecord, currentcandidates, cg, ts, i)
        currentcandidates = []
        if len(new_c_stars) >= 1:
            for c in new_c_stars:
                matching_chunks.add(c)
                c.parse = c.parse + 1
                currentcandidates = currentcandidates + list(c.cl.keys())
                print()
            c_stars = new_c_stars
            c_stars_T = new_c_stars_T
            c_stars_delta_i = new_c_stars_delta_i
                #l = sum([len(list(item)) for item in c_stars.ordered_content])
        else: break # no more match from the left chunk

    # find the biggest matching chunk:
    for c, cT, c_delta_i in zip(c_stars, c_stars_T, c_stars_delta_i):
        if cT >= maxl:
            maxchunk = c
            maxchunkcontent = c.key if type(c) == Variable else c.content
            max_delta_i = c_delta_i
            maxl = cT
    return maxchunk, maxchunkcontent, maxl, max_delta_i  # return the biggest chunk and the remaining sequence

def check_meta_match(chunkrecord, currentcandidates, cg, ts, i):
    '''check whether any or many of the currentcandidates are matching with the chunkrecord at position i
    :param chunkrecord: the chunk record of the sequence
    :param currentcandidates: the current candidates to check whether it is matching with the chunkrecord at position i
    :param cg: the chunk graph
    :param i: the index of the sequence that ck ends at
    :param ts: the list of ending time of the chunk record

    :return:
        match: the best chunk that is consistent with the sequence
    '''
    matching_chunks = []
    matching_Ts = []
    matching_delta_i = []
    if i >= len(ts): return False, matchingcontent, matching_delta_i

    for chunk in currentcandidates: # iterate through ordered_content of chunks
        current_t = ts[i]
        match = True# iterate through chunk content to check consistency with concrete observation
        matching_content = []# only when all of the chunk's ordered content matches with chunkrecord, then it is a match
        match_t = 0
        match_delta_i = len(chunk.ordered_content)

        for j in range(0, len(chunk.ordered_content)):  # need to add pre-existing components
            singlecontent = chunk.ordered_content[j]
            if i+j >= len(ts):
                match = False
                break
            t = ts[i+j]
            if type(singlecontent) == set:# concrete chunks
                if singlecontent == set(chunkrecord[t][0][0]):  # concrete chunks
                    match_t = match_t + len(singlecontent)
                    matching_content.append(chunkrecord[t][0][0])
                else:
                    match = False
                    break # try the next candidate
            else: # variable
                assert type(singlecontent) == str
                seqkey = chunkrecord[t][0][0] # what is actually in the sequence
                seqchunk = cg.chunks[seqkey] if seqkey in cg.chunks else cg.variables[seqkey]
                if singlecontent in seqchunk.abstraction or singlecontent == seqkey:
                    cg.variables[singlecontent].T = seqchunk.T
                    match_t = match_t + seqchunk.T
                    matching_content.append(singlecontent)
                else:
                    match = False
                    break

        if ~match: # also check if chunkkey matches the chunkrecord
            t = ts[i]
            if chunk.key == chunkrecord[t][0][0]:
                match_t = chunk.T
                matching_content = [chunk.key]
                match = True
                match_delta_i = 1

        if match:
            chunk.T = match_t # update the temporal length of the chunk (since there might be variables inside)
            matching_chunks.append(chunk)
            matching_Ts.append(match_t)
            matching_delta_i.append(match_delta_i)

    return matching_chunks, matching_Ts, matching_delta_i

def rational_learning(cg, n_update=10, complexity_limit=-np.log2(0.10)):
    """ given a learned representation, update chunks based on rank of joint occurrence frequency and hypothesis tests
            Parameters:
                n_update: the number of concatinations made based on the pre-existing cg records
                complexity_limit:
            Returns:
                cg: chunking graph with empty chunks
    """
    totalcount = sum([item.count for item in list(cg.chunks.values())])

    candidancy_pairs = (
        []
    )
    # check every chunk pair in the transition matrix and come up with a set of new chunks to update
    for _previdx, _prevck in cg.chunks.items() | cg.variables.items():  # this iteration will be slow.
        for _postidx in _prevck.adjacency:
            for _dt in _prevck.adjacency[_postidx].keys():
                if _postidx in cg.chunks:
                    _postck = cg.chunks[_postidx]
                else:
                    _postck = cg.variables[_postidx]
                # hypothesis test
                if (
                        cg.hypothesis_test(_previdx, _postidx, _dt) == False
                ):  # reject null hypothesis
                    _cat = combinechunks(_previdx, _postidx, _dt, cg)
                    _cat_count = _prevck.adjacency[_postidx][_dt]
                    if _cat is not None:
                        p_c = _cat_count/totalcount
                        storage_cost = -np.log(p_c)*p_c
                        L_c = _prevck.T + _postck.T
                        ER = L_c*p_c/cg.learning_data[-1][3] # log explanatory ratio
                        candidancy_pairs.append(
                            [
                                (_previdx, _postidx, _cat, _dt),
                                _prevck.adjacency[_postidx][_dt], ER/storage_cost
                            ]
                        )

    candidancy_pairs.sort(key=lambda tup: tup[2], reverse=True)

    # number of chunk combinations allowed.
    cumcomplexity = 0 #cumulative complexity
    ratio = 0.7 # ER/storage_cost
    for i in range(0, min(n_update, len(candidancy_pairs))):
        if candidancy_pairs[i][-1]<ratio:
            break

        prev_idx, current_idx, cat, dt = candidancy_pairs[i][0]
        cg.chunking_reorganization(prev_idx, current_idx, cat, dt)
        complexity = -np.log2(candidancy_pairs[i][1] / totalcount)
        if complexity < 0: print()
        print('complexity is ', complexity)
        cumcomplexity = cumcomplexity + complexity
        if cumcomplexity > complexity_limit:
            print('reaching complexity limit of ', complexity_limit)
            break
        if i > len(candidancy_pairs):break
    return cg, candidancy_pairs


def loadchunkintoarray(chunkcontent):
    H = 1
    W = 1
    chunkarray = np.zeros((int(max(np.atleast_2d(np.array(list(chunkcontent)))[:, 0]) + 1), H, W))
    for t, i, j, v in list(chunkcontent):
        chunkarray[t, i, j] = int(v)
    return chunkarray


def sample_transition_length_bias(cg, prev, length_bias=10):
    if len(prev.adjacency.keys()) == 0:
        return sample_marginal_length_bias(cg, length_bias=10)
    else:
        prob = []
        states = []
        maxlc = 0
        maxlci = 0
        T = 1 # set time length to 1 by default
        for i in range(0, len(list(prev.adjacency.keys()))):
            ckidx = list(prev.adjacency.keys())[i]
            if ckidx in cg.chunks:
                chunk = cg.chunks[ckidx]
                T = chunk.T
            else: chunk = cg.variables[ckidx]
            prob.append(prev.adjacency[ckidx][0])  # transition frequencies
            states.append(chunk)
            if T > maxlc:
                maxlci = i
                maxlc = T
        prob[maxlci] = prob[maxlci] + length_bias  # prioritize maximal length
        prob = [k / sum(prob) for k in prob]

        return sample_from_distribution(states, prob)


def content_to_set(ordered_content):
    # return the variable instantiated content
    content_set = set()
    tshift = 0
    for content in ordered_content:
        maxdt = 0
        for signal in content:
            if signal[0] >= maxdt:
                maxdt = signal[0]

            tshiftsignal = list(signal).copy()
            tshiftsignal[0] = tshiftsignal[0] + tshift
            content_set.add(tuple(tshiftsignal))

        tshift = tshift + maxdt + 1
    return content_set


def recall(cg, seql=12, firstitem=1):
    ''' Sequence recall based on the priming of the first item,
    and recall subsequent items based on transition sampling'''
    img = np.zeros([1, 1, 1])
    ps = []
    l = 0
    # sample the first recalled item
    consistent = False
    while consistent == False:
        chunk, p = sample_marginal_length_bias(cg)
        if len(chunk.includedvariables) > 0 or chunk.content == None:
            cg.sample_variable_instances(generative_model=False)  # instantiate variables in cg into concrete instances
            sampled_content = cg.obtain_concrete_content_from_variables(chunk.key)
            chunk.content = content_to_set(sampled_content)
        if loadchunkintoarray(chunk.content)[0, 0, 0] - firstitem == 0:
            consistent = True

    img = np.concatenate((img, loadchunkintoarray(chunk.content)), axis=0)
    ps.append(p)

    l = l + len(chunk.content)
    prev = chunk

    while l < seql:
        chunk, p = sample_transition_length_bias(cg, prev)
        if type(chunk) == Chunk:
            cg.sample_variable_instances(
                generative_model=False)  # instantiate variables in cg into concrete instances
            sampled_content = cg.obtain_concrete_content_from_variables(chunk.key)
            chunk.content = content_to_set(sampled_content)
        else:
            cg.sample_variable_instances(
                generative_model=False)  # instantiate variables in cg into concrete instances
            sampled_content = cg.obtain_concrete_content_from_variables(chunk.key)
            chunk.content = content_to_set(sampled_content)

        img = np.concatenate((img, loadchunkintoarray(chunk.content)), axis=0)
        prev = chunk
        l = l + len(chunk.content)
        ps.append(p)

    # print(img[1:seql, :, :])
    return img[1:seql + 1, :, :], ps


def sample_marginal_length_bias(cg, length_bias=10):
    # sample subsequences while prioritize maximal length recall
    prob = []
    states = []
    maxlc = 0
    maxlci = 0
    for i in range(0, len(list(cg.chunks))):
        chunk = list(cg.chunks.values())[i]
        prob.append(chunk.count)
        states.append(chunk)
        if chunk.content == None:
            if len(chunk.ordered_content) > maxlc:
                maxlci = i
                maxlc = len(chunk.ordered_content)
        else:
            if len(chunk.content) > maxlc:
                maxlci = i
                maxlc = len(chunk.content)

    prob[maxlci] = prob[maxlci] + length_bias  # prioritize maximal length
    prob = [k / sum(prob) for k in prob]

    return sample_from_distribution(states, prob)


def sample_from_distribution(states, prob):
    """
    states: a list of chunks
    prob: another list that contains the probability"""
    prob = [k / sum(prob) for k in prob]
    cdf = [0.0]
    for s in range(0, len(states)):
        cdf.append(cdf[s] + prob[s])
    k = np.random.rand()
    for i in range(1, len(states) + 1):
        if (k >= cdf[i - 1]):
            if (k < cdf[i]):
                return states[i - 1], prob[i - 1]


def abstraction_update(current_chunks, chunk_record, cg, t, freq_T = 6):
    """
    Learn abstraction online based on recently encounterd sequential elements.
    Create variables from adjacency matrix.
    variable construction: chunks that share common ancestors and common descendents.
    pre---variable---post, for each dt time: variables with common cause and common effect
    freq_T: frequency threshold
    """
    # TODO: another version with more flexible dt
    varchunks_to_add = []
    T = 3
    n_t = 1
    if len(chunk_record) > 2:  # update transition only when there are more than 2 observed units
        for chunkname in current_chunks:  # latestdescendents
            chunk = cg.chunks[chunkname]
            d_t = 1  # the difference between the end of the current chunk and the end of the previous chunks
            temporal_length_chunk = chunk.T
            v_vertical_ = set(chunk.preadjacency.keys())
            while d_t <= temporal_length_chunk + n_t and len(
                    chunk_record) > 1:  # looking backward to find, padding length

                # adjacent chunks
                chunkendtime = t - d_t
                if chunkendtime in list(chunk_record.keys()) and chunkendtime + temporal_length_chunk != t:  # exclude immediate preceeding chunk before the current chunk
                    previous_chunks = chunk_record[chunkendtime]
                    # current_chunk_starting_time = t - temporal_length_of_current_chunk + 1  # the "current chunk" starts at:
                    for prevname, _ in previous_chunks:
                        if not (prevname == chunkname and d_t == 0):
                            prev = cg.chunks[prevname]
                            v_horizontal_ = set(prev.adjacency.keys())
                            temp_variable_entailment = v_horizontal_.intersection(v_vertical_)
                            candidate_variable_entailment = set()
                            freq_c = 0
                            for c in temp_variable_entailment:
                                if chunk.preadjacency[c][0] > 0:
                                    candidate_variable_entailment.add(c)
                                freq_c = freq_c + chunk.preadjacency[c][0]

                            candidate_variable_entailment = cg.filter_entailing_variable(candidate_variable_entailment)
                            if len(candidate_variable_entailment) >= T and freq_c > freq_T:  # register a variable
                                print('previous chunk: ', prev.key, ' post chunk: ', chunk.key,
                                      ' candidate variable entailment ', temp_variable_entailment, 'freq', freq_c)
                                candidate_variables = {}
                                for candidate in candidate_variable_entailment:
                                    if candidate in cg.chunks:candidate_variables[candidate] = cg.chunks[candidate]
                                    else: candidate_variables[candidate] = cg.variables[candidate]

                                v = Variable(candidate_variables)
                                v = cg.add_variable(v, candidate_variable_entailment)
                                # create variable chunk: chunk + var + postchunk
                                # need to roll it out when chunk itself contains variables.
                                ordered_content = prev.ordered_content.copy()
                                ordered_content.append(v.key)
                                ordered_content = ordered_content + chunk.ordered_content
                                V = {}
                                V[v.key] = v
                                var_chunk = Chunk(([]), includedvariables=V, ordered_content=ordered_content)
                                var_chunk.count = freq_c
                                varchunks_to_add.append(var_chunk)
                                #############################################

                                prev.cl = cg.check_and_add_to_dict(prev.cl, var_chunk)
                                chunk.cr = cg.check_and_add_to_dict(chunk.cr, var_chunk)
                                var_chunk.acl = cg.check_and_add_to_dict(var_chunk.acl, prev)
                                var_chunk.acr = cg.check_and_add_to_dict(var_chunk.acr, chunk)

                d_t = d_t + 1

    for var_chunk in varchunks_to_add:
        cg.add_chunk(var_chunk)
    print('the number of newly learned variable chunk is: ', len(varchunks_to_add))
    return cg


def learning_and_update(current_chunks, chunk_record, cg, t, threshold_chunk=False):
    '''
    Update transitions and marginals and decide to chunk
    t: finishing parsing at time t
    current_chunks_idx: the chunks ending at the current time point
    chunk_record: boundary record of when the previous chunks have ended.
    threshold_chunk: False when the function is used only for parsing and updating transition probabilities. '''
    n_t = 0  # 2
    if len(chunk_record) > 1:  # update transition only when there are more than 2 observed units
        for chunk in current_chunks:
            d_t = 0  # the difference between the end of the current chunk and the end of the previous chunks
            if chunk in cg.chunks: temporal_length_chunk = cg.chunks[chunk].T
            else: temporal_length_chunk = cg.variables[chunk].T
            while d_t <= temporal_length_chunk + n_t:  # looking backward to find
                chunkendtime = t - d_t
                if chunkendtime in list(chunk_record.keys()):
                    previous_chunks = chunk_record[chunkendtime] #current_chunk_starting_time = t - temporal_length_of_current_chunk + 1  # the "current chunk" starts at:
                    for prev, _ in previous_chunks:
                        if not (prev == chunk and d_t == 0):
                            pre_c = cg.chunks[prev] if prev in cg.chunks else cg.variables[prev]
                            pos_c = cg.chunks[chunk] if chunk in cg.chunks else cg.variables[chunk]
                            adjacent, dt = adjacency(pre_c, pos_c, d_t, t, cg)
                            if dt != 0:
                                print('previous chunk: ', prev, ' post chunk: ', chunk, ' dt ', dt)
                            if adjacent:
                                pre_c.update_transition(pos_c, dt, variable_adjacency_update = True) # update adjacency for both chunks and variables
                                if threshold_chunk:
                                    combined_chunk = combinechunks(prev, chunk, dt, cg)
                                    cg, chunked = threshold_chunking(prev, chunk, combined_chunk, dt, cg, N = 6)
                d_t = d_t + 1
    return cg


def threshold_chunking(prev_key, current_key, combined_chunk, dt, cg, N = 6):
    """combined_chunk: a new chunk instance
    dt: end_prev(inclusive) - start_post(exclusive)"""

    """cg: chunking graph
    learning function manipulates when do the chunk update happen
    chunks when some combination goes beyond a threshold"""
    chunked = False  # unless later it was verified to fit into the chunking criteria.
    cat = combined_chunk
    prev = cg.chunks[prev_key]
    current = cg.chunks[current_key]
    if current_key in prev.adjacency and dt in prev.adjacency[current_key]:
        if prev.count > N and prev.adjacency[current_key][dt] > N:
            hypothesis_result = cg.hypothesis_test(prev_key, current_key, dt)
            if hypothesis_result is False:
                chunked = True
                print(cat.ordered_content)
                cg.chunking_reorganization(prev_key, current_key, cat, dt)

    # also check the case with prev + variable(of current)
    for ak, av in current.abstraction.items():
        if ak in prev.adjacency and dt in prev.adjacency[ak]:
            if prev.count > N and prev.adjacency[ak][dt] > N:
                hypothesis_result = cg.hypothesis_test(prev_key, ak, dt)
                ordered_content = prev.ordered_content.copy()
                ordered_content.append(ak)
                V = {}
                V[ak] = av
                cat = Chunk(([]), includedvariables=V, ordered_content=ordered_content)
                if hypothesis_result is False:
                    chunked = True
                    #print(cat.ordered_content)
                    cg.chunking_reorganization(prev_key, ak, cat, dt)

    return cg, chunked


import itertools


def check_overlap(array1, array2):
    output = np.empty((0, array1.shape[1]))
    for i0, i1 in itertools.product(np.arange(array1.shape[0]),
                                    np.arange(array2.shape[0])):
        if np.all(np.isclose(array1[i0], array2[i1])):
            output = np.concatenate((output, [array2[i1]]), axis=0)
    return output


def checkequal(chunk1, chunk2):
    if len(chunk1.content.intersection(chunk2.content)) == max(len(chunk1.content), len(chunk2.content)):
        return True
    else:
        return False


def evaluatesimilarity(chunk1, chunk2):
    return chunk1.checksimilarity(chunk2)


def adjacency(prev, post, time_diff, t, cg):
    ''' returns empty matrix if not chunkable '''
    if prev.key == post.key and time_diff == 0:  # do not chunk a chunk by itself.
        return False, -100
    else:
        e_post = t  # inclusive
        e_prev = t - time_diff  # inclusive
        s_prev = e_prev - prev.T  # exclusive
        s_post = e_post - post.T  # exclusive
        dt = e_prev - s_post
        delta_t = e_prev - max(s_post, s_prev)  # the overlapping temporal length between the two chunks
        if delta_t >= 0:
            return True, dt
        else:
            return False, dt




def combinechunks(prev_key, post_key, dt, cg):
    # time_diff: difference between end of the post chunk and the end of the previous chunk
    ''' returns empty matrix if not chunkable '''
    # update transitions between chunks with a temporal proximity
    # chunk ends at the point of the end_point_chunk
    # candidate chunk ends at the point of the end_point_candidate_chunk
    if prev_key in cg.variables or post_key in cg.variables:
        Var = dict()
        if prev_key in cg.variables:
            prev = cg.variables[prev_key]
            Var[prev_key] = prev
        else:
            prev = cg.chunks[prev_key]
            Var = Var | prev.includedvariables

        if post_key in cg.variables:
            post = cg.variables[post_key]
            Var[post_key] = post
        else:
            post = cg.chunks[post_key]
            Var = Var | post.includedvariables

        # combine chunks by merging ordered content
        T = 0
        combinedcontent = []
        for item in prev.ordered_content:
            combinedcontent.append(item)
        T = T + prev.T

        for item in post.ordered_content:
            combinedcontent.append(item)  # shifted content representation
        T = T + post.T
        combinedchunk = Chunk(list([]), includedvariables = Var, ordered_content=combinedcontent)
        combinedchunk.T = T
        return combinedchunk

    else:
        post = cg.chunks[post_key]
        e_prev = 0
        l_t_post = post.T
        s_post = e_prev - dt  # start point is inclusive
        e_post = s_post + l_t_post
        if prev_key == post_key and e_post == e_prev:
            return None, -100  # do not chunk a chunk by itself.
        else:
            # TODO: double check if combine chunks and check adjacency generates the same chunk agglomeration
            prev = cg.chunks[prev_key]
            post = cg.chunks[post_key]
            e_prev = 0
            l_t_prev = prev.T
            l_t_post = post.T
            s_prev = e_prev - l_t_prev  # the exclusive temporal length
            s_post = e_prev - dt  # start point is inclusive
            e_post = s_post + l_t_post
            # dt = e_prev - s_post
            delta_t = e_prev - max(s_post, s_prev)  # the overlapping temporal length between the two chunks
            t_chunk = max(e_post, e_prev) - min(e_post, e_prev) + delta_t + max(s_post, s_prev) - min(s_post,
                                                                                                      s_prev)  # the stretching temporal length of the two chunks

            if t_chunk == l_t_prev and t_chunk == l_t_post and checkequal(prev, post):
                return None  # do not chunk a chunk by itself.
            else:
                # initiate a new chunk.
                if prev.content == None or post.content == None:
                    return prev.concatinate(post, check = False)
                else:
                    prevcontent = prev.content.copy()
                    postcontent = post.content.copy()

                    if s_prev > s_post:  # post start first
                        # shift content in time:
                        prev_shift = s_prev - s_post
                        newprevcontent = set()
                        for msrm in prevcontent:
                            lmsrm = list(msrm)
                            lmsrm[0] = lmsrm[0] + prev_shift  # pad the time dimension
                            newprevcontent.add(tuple(lmsrm))
                    prevchunk = Chunk(list(prevcontent), H=prev.H, W=prev.W)
                    prevchunk.T = prev.T
                    prevchunk.H = prev.H
                    prevchunk.W = prev.W

                    if s_prev < s_post:  # prev start first
                        post_shift = s_post - s_prev
                        newpostcontent = set()
                        for msrm in postcontent:
                            lmsrm = list(msrm)
                            lmsrm[0] = lmsrm[0] + post_shift
                            newpostcontent.add(tuple(lmsrm))
                        postcontent = newpostcontent
                    postchunk = Chunk(list(postcontent), H=post.H, W=post.W)
                    postchunk.T = post.T
                    postchunk.H = post.H
                    postchunk.W = post.W
                return prevchunk.concatinate(postchunk)


def add_singleton_chunk_to_M(seq, t, cg):
    '''Add the singleton chunk observed at the temporal point t of the seq to the set of chunks M'''
    H, W = seq.shape[1:]
    current_chunks = set()  # the set of chunks that end right before this moment t.
    for i in range(0, H):
        for j in range(0, W):
            chunk = np.zeros([1, H, W])
            chunk[0, i, j] = seq[t, i, j]
            if np.sum(np.abs(chunk)) > 0:  # not an empty chunks
                matchingchunk = cg.add_chunk_to_cg_class(chunk)
                current_chunks.add(matchingchunk)  # the chunk that ends right at the moment t
    return cg, current_chunks


def check_chunk_exist(chunk, M):
    for ck in M:
        if ck.profile == chunk:
            return True
    return False


def add_chunk_to_M(chunk, M):
    if chunk in list(M.keys()):
        M[chunk] = M[chunk] + 1
    else:
        M[chunk] = 1
    return M


def find_relevant_observations(seq, termination_time, t):
    H, W = seq.shape[1:]
    min_termin = min(termination_time.flatten())
    relevant_observations = np.zeros([int(t - min(termination_time.flatten())), H, W])
    for i in range(0, H):
        for j in range(0, W):
            relevant_t = (termination_time[i, j] - min_termin)
            relevant_observations[relevant_t:, i, j] = seq[termination_time[i, j] + 1:, i, j]
    relevant_observations, n_zero = refactor_observation(relevant_observations, return_max_k=True)  # cut such that
    # there is an event
    # happening
    # at the beginning.
    return relevant_observations, n_zero


# no chunks ends at this moment
def update_termination_time(current_chunks, chunk_termination_time, t):
    """For each spatial dimension, update the time that the latest chunk terminated
            :returns
                chunk_termination_time: shape: np.array(H,W), the height and width of the spatial dimension, in
                each dimension, the latest time that a chunk has ended. """
    for chunk in current_chunks:  # update chunk termination time for the current chunks
        chunk_termination_time[np.array(chunk)[-1, :, :] > 0] = t
    return chunk_termination_time


def identify_chunks(cg, seq, termination_time, t, previous_chunk_boundary_record):
    """Identify chunks to explain the current observation, and
    if the pre-existing chunk does not explain the current observation
    create new chunks
    # which one of them is predicted, which one of them is surprising.
    cg: chunking graph object
    seq: sequence of observation
    termination time: the time in each dimension where the last chunk has terminated.
    """
    "Starting from the location where the last chunk terminates, " \
    "needs explanation for what comes thereafter "
    # TODO: space for optimization to not parse the entire sequence
    relevant_observations, nzero = find_relevant_observations(seq, termination_time, t)
    # cut such that there is an event

    # happening at the beginning.
    if relevant_observations.shape[0] > 0:  # there are some observations to explain:
        # use the chunks to explain the current relevant observations.
        observation_to_explain = relevant_observations.copy()  # the sequence the same size as chunks:
        if len(list(cg.M.keys())) > 0:
            # check chunk match, start from the biggest chunk/the most likely
            # in the order of temporal slice, explain the observation one by one, until the entire observation is
            # cancelled
            # load things into current chunks
            current_chunks, termination_time, cg = identify_chunk_ending_with_M(observation_to_explain, cg,
                                                                                termination_time, t,
                                                                                previous_chunk_boundary_record)
            previous_chunk_boundary_record.append(current_chunks)
        else:  # nothing in the marginals, find the singleton chunks and load the marginals, relevant obsertvation size
            # = 1
            cg, current_chunks = add_singleton_chunk_to_M(seq, t, cg)
            delta_t = (tuple_to_arr(observation_to_explain).shape[0] - 1)
            t_end_best_matching_subchunk = t - delta_t
            termination_time = update_termination_time(current_chunks, termination_time, t_end_best_matching_subchunk)
            previous_chunk_boundary_record.append(current_chunks)
        # moved transition update to learning and update
        return current_chunks, cg, termination_time, previous_chunk_boundary_record
    else:
        return {}, cg, termination_time, previous_chunk_boundary_record


def pop_chunk_in_seq_full(chunkcontent, seqc, cg):  # _c_
    for tup in chunkcontent:  # remove chunk content one by one from the sequence
        seqc.remove(tup)  # O(1)
    return seqc


def check_chunk_in_seq(chunk, seq):  # _c_
    """ returns the size of match, how big the matching content is, and how big the chunk is, to be matched"""
    # returns the volume and the temporal length of the chunk content
    content = chunk.content
    if content.issubset(seq):
        return chunk.volume, chunk.T
    else:
        return 0, 0


def check_chunk_in_seq_boundary(chunk, seq):
    ''' Use chunk boundaries to identify chunks in the sequence
        Discrepencies of none-boundary contents are being ignored '''
    boundary = chunk.boundarycontent
    assert boundary is not None
    if boundary.issubset(seq):
        return chunk.volume, chunk.T
    else:
        return 0, 0


# compute the absolute difference between the observation and the template pixel

def check_chunk_in_seq_approximate(chunk, seq):  # more approximate
    '''TODO: Calculate the minimal approximate distance between chunk and sequence observations'''
    if chunk.check_match(seq):
        return chunk.volume, chunk.T
    else:
        return 0, 0


def pop_chunk_in_seq_boundary(chunk_idx, seqc, cg):
    # pop everything within the boundary of this chunk in seqc
    chunk = cg.chunks[chunk_idx]
    content = list(chunk.boundarycontent)
    T = sorted(content, key=lambda tup: tup[0])
    X = sorted(content, key=lambda tup: tup[1])
    Y = sorted(content, key=lambda tup: tup[2])

    tmax, tmin = T[-1][0], T[0][0]
    xmax, xmin = X[-1][1], X[0][1]
    ymax, ymin = Y[-1][2], Y[0][2]
    seqcc = seqc.copy()
    for tup in seqc:  # remove chunk content one by one from the sequence
        if tup[0] >= tmin and tup[0] <= tmax and tup[1] >= xmin and tup[1] <= xmax and tup[2] >= ymin and tup[
            2] <= ymax:
            seqcc.remove(tup)
            if tup not in chunk.content:
                chunk.content.add(tup)  # add sequential content within a chunk
    return seqcc


def ordered_content_to_sequence(ordered_content):
    """Convert a list of sets in ordered_content into a set
    *checked*"""
    content_set = set()
    content_set = content_set | ordered_content[0]
    T = 0
    for i in range(1, len(ordered_content)):  # need to add pre-existing components
        prevcontent = ordered_content[i - 1]
        T = T + sorted(list(prevcontent))[-1][0] + 1 if len(prevcontent) > 0 else 0  # maximal finishing time
        currentcontent = ordered_content[i]
        content_set = content_set | timeshift(currentcontent, T)
    return content_set


def timeshift(content, t):
    shiftedcontent = []
    for tup in list(content):
        lp = list(tup)
        lp[0] = lp[0] + t
        shiftedcontent.append(tuple(lp))
    return set(shiftedcontent)


def check_recursive_match(seqc, matchingcontent, chunk, cg):
    ''' when called, matching content is initialized to be set(), seqc: a copy of the matching sequence,
       iterate through chunk content to check consistency with concrete observation '''
    match = True
    if len(seqc) < len(chunk.ordered_content): return False, matchingcontent
    for ck in chunk.ordered_content:  # need to add pre-existing components
        if type(ck) == set:  # concrete chunks
            content = ck
            T = sorted(list(matchingcontent))[-1][0] + 1 if len(matchingcontent) > 0 else 0  # maximal finishing time
            temp = matchingcontent.copy()
            temp = temp.union(timeshift(content, T))
            if temp.issubset(seqc):
                matchingcontent = temp
            else: # not a match, exit
                return False, temp
        else: # variable
            assert type(ck) == str
            vck = cg.variables[ck]
            entailingchunkmatching = []
            entaillingchunkmatchingcontent = []
            entailingchunksize = []
            maxsize = 0
            maxcontent = None
            if type(vck.entailingchunks) == set:
                temp = list(vck.entailingchunks)
            else:
                temp = vck.entailingchunks.values()

            for v in temp:  # chunks inside a variable
                match, content = check_recursive_match(seqc, matchingcontent, v, cg)
                entailingchunkmatching.append(match)
                entaillingchunkmatchingcontent.append(content)
                entailingchunksize.append(len(content))
                if match:
                    if len(content) > maxsize:
                        maxsize = len(content)
                        maxcontent = content
            if any(entailingchunkmatching) == False: match = False
            else: # find entaillingchunkmatchingcontent with the maximal size
                match = True
                matchingcontent = maxcontent
    return match, matchingcontent


def identify_biggest_chunk(cg, seqc, candidate_set):  # _c_full
    # variables: the set of variable with their corresponding (associated chunk) to parse the sequence
    '''Chunk with a bigger size is priorized to explain the sequence'''
    if len(candidate_set) == 0: candidate_set = set(cg.chunks.values())

    def findmatch(current, seqc):
        maxmatchingcontent = set()
        maxchunk = None
        for chunk in current:  # can be multiple chunks to fit the sequence
            match, matchingcontent = check_recursive_match(seqc, set(), chunk, cg)
            if match:
                if len(matchingcontent) > len(maxmatchingcontent):
                    maxmatchingcontent = matchingcontent
                    maxchunk = chunk
        if maxchunk is not None:
            return maxchunk, maxmatchingcontent
        else:
            return None, None

    pop_chunk_in_seq = pop_chunk_in_seq_full
    maxchunk = None
    maxchunkcontent = None
    maxl = 0
    current = cg.ancestors  # start searching for matching chunks from the ancestors
    matching_chunks = set()  # all chunks that are consistent with the sequence

    if len(cg.ancestors) == 0: maxchunk = None
    else:
        while len(current) > 0:  # search until the bottom of the parsing tree from abstract to concrete
            c_star, c_star_content = findmatch(current, seqc)
            if c_star is not None:
                matching_chunks.add(c_star)
                c_star.parse = c_star.parse + 1
                current = list(c_star.cl.keys())
                maxchunk = c_star
                maxchunkcontent = c_star_content
                maxl = len(c_star_content)
                maxchunk.T = len(c_star_content)
                c_star.T = len(c_star_content)
            else: break

    if maxchunk is None: # no existing representation is identified
        maxchunk, cg = identify_singleton_chunk(cg, seqc)
        maxchunkcontent = maxchunk.ordered_content[0]
        maxl = len(maxchunkcontent)
        maxchunk.T = maxl
        seqc = pop_chunk_in_seq(maxchunkcontent, seqc, cg)  # pop identified chunks in sequence
    else: # maxchunk is not None #eligible_chunks = list(matching_chunks.intersection(candidate_set))
        seqc = pop_chunk_in_seq(maxchunkcontent, seqc, cg)  # pop identified chunks in sequence
    maxchunk.content = maxchunkcontent

    return maxchunk, seqc, maxl  # return the biggest chunk and the remaining sequence


def identify_singleton_chunk(cg, seqc):  # _c_
    chunkcontent = [seqc[0]]
    chunk = Chunk(chunkcontent, H=cg.H, W=cg.W)
    cg.add_chunk(chunk, ancestor=True)
    return chunk, cg


def updatechunk(chunk, explainchunk, chunk_record, cg, t, maxl=0):
    explainchunk.append((chunk.key, chunk.T))
    cg.chunks[chunk.key].update()  # update chunk count among the currently identified chunks
    if len(chunk.includedvariables) == 0: # no variables inside the chunk
        timespan = int(chunk.T) + t - 1
    else: timespan = maxl + t - 1
    chunk_record, update = updatechunkrecord(chunk_record, chunk.key, timespan, cg)
    return explainchunk, cg, chunk_record, update


def updatechunkabstract(chunk, abstract_chunk_record, cg, endindex, ts, maxl):
    """update the abstract chunk record
    maxl: temporal length of the chunk"""

    if chunk.key in cg.chunks:
        cg.chunks[chunk.key].update()  # update chunk count among the currently identified chunks
        cg.chunks[chunk.key].T = maxl
    else:
        cg.variables[chunk.key].update()
        cg.variables[chunk.key].T = maxl
    endtime = ts[endindex] # either endindex or endindex - 1
    if endtime not in list(abstract_chunk_record.keys()):
        if chunk.key in cg.chunks:
            abstract_chunk_record[endtime] = [(chunk.key, cg.chunks[chunk.key].count)]
        else:
            abstract_chunk_record[endtime] = [(chunk.key, cg.variables[chunk.key].count)]
    else:
        if chunk.key not in abstract_chunk_record[endtime]:  # one chunk can only be identified at one time point. when there are
            # multiple points that correspond to the same chunk, this chunk is identified as occurring once.

            if chunk.key in cg.chunks:
                abstract_chunk_record[endtime].append((chunk.key, cg.chunks[chunk.key].count))
            else:
                abstract_chunk_record[endtime].append((chunk.key, cg.variables[chunk.key].count))
    return cg, abstract_chunk_record



def identify_one_chunk(cg, seqc, explainchunk, chunk_record, t, candidate_set, PRINT = False):  # _c_
    max_chunk, seqc, maxl = identify_biggest_chunk(cg, seqc, candidate_set)  # identify and pop chunks in sequence
    explainchunk, cg, chunk_record, newupdate = updatechunk(max_chunk, explainchunk, chunk_record, cg, t, maxl=maxl)
    if PRINT:
        print('max chunk is ', max_chunk.key, 't is ', t, 'max chunk length is ', max_chunk.T, 'maxl is, ', maxl)
        print('time span of the max chunk is', max_chunk.T)
        print('explain_chunk key is', explainchunk[0][0])
        print(newupdate)
    # explainchunk.append((max_chunk_idx, int(cg.chunks[max_chunk_idx].T), list(cg.visible_chunk_list[max_chunk_idx])))
    # chunk_record = updatechunkrecord(chunk_record, max_chunk_idx, int(cg.chunks[max_chunk_idx].T) + t, cg)
    # cg.chunks[max_chunk_idx].update()  # update chunk count among the currently identified chunks
    return cg, seqc, chunk_record, explainchunk


def pop_chunk_in_seq_approximate(chunk_idx, seqc, cg):
    # pop everything within the boundary of this chunk in seqc
    chunk = cg.chunks[chunk_idx]
    print(' chunk content before ', chunk.content)
    print(' matching seq ', chunk.matching_seq)

    chunk.average_content()  # average content
    print(' chunk content after ', chunk.content)

    cg.visible_chunk_list[chunk_idx] = chunk.content
    seqcc = seqc.copy()
    for m in list(chunk.matching_seq.keys()):
        for pt in chunk.matching_seq[m]:
            # can be the case that the same point in sequence matches multiple keys
            try:
                seqcc.remove(pt)
            except(ValueError):
                print(' one point matches two chunk points ')

    return seqcc


def identify_one_chunk_approximate(cg, seqc, explainchunk, chunk_record, t):
    maxsizematch = 0
    maxchunk = None
    for chunk in reversed(cg.chunks):  # identify one chunk in sequence at one time...
        matchsize, match_len = check_chunk_in_seq_approximate(chunk, seqc)
        if matchsize > maxsizematch:
            maxsizematch = matchsize
            maxchunk = chunk.index

    # remove chunk from seq
    if maxchunk is None:
        maxchunk, cg = identify_singleton_chunk(cg, seqc)
        seqc = pop_chunk_in_seq_full(maxchunk, seqc, cg)
    else:
        seqc = pop_chunk_in_seq_approximate(maxchunk, seqc, cg)

    max_chunk_idx = maxchunk
    explainchunk.append((max_chunk_idx, int(cg.chunks[max_chunk_idx].T), list(cg.visible_chunk_list[max_chunk_idx])))
    chunk_record = updatechunkrecord(chunk_record, max_chunk_idx, int(cg.chunks[max_chunk_idx].T) + t, cg)
    cg.chunks[maxchunk.key].update()  # update chunk count among the currently identified chunks
    return cg, seqc, chunk_record, explainchunk


def check_seq_explained(seqc):  # _c_
    # check whether there is a t = 0 in the seqc:
    if seqc == []:
        return True
    else:
        return seqc[0][0] != 0  # finished explaining the current time point


def updatechunkrecord(chunk_record, ckidx, endtime, cg, freq=True):
    if freq == True:
        if endtime not in list(chunk_record.keys()):
            chunk_record[endtime] = [(ckidx, cg.chunks[ckidx].count)]
        else:
            if ckidx not in chunk_record[endtime]:  # one chunk can only be identified at one time point. when there are
                # multiple points that correspond to the same chunk, this chunk is identified as occurring once.
                chunk_record[endtime].append((ckidx, cg.chunks[ckidx].count))
    else:
        p = cg.chunks[ckidx].count / endtime
        if endtime not in list(chunk_record.keys()):
            chunk_record[endtime] = [(ckidx, p)]
        else:
            if ckidx not in chunk_record[endtime]:  # one chunk can only be identified at one time point. when there are
                # multiple points that correspond to the same chunk, this chunk is identified as occurring once.
                chunk_record[endtime].append((ckidx, p))
    return chunk_record, (endtime, ckidx)


def identify_latest_chunks(cg, seq, chunk_record, t, candidate_set, PRINT = True):
    ''' use the biggest explainable chunk to parse the sequence and store chunks in the chunkrecord
        candidate_set: the specified set of variables '''
    # identify biggest chunks that finish at the earliest time point.
    t = t + 1  # the current time (buffer updates after parsing)
    seq_explained = False
    seqc = seq.copy()
    explainchunk = []# find candidate chunks explaining the upcoming sequence, out of these candidate chunks, find the ones that terminates at the earliest time point.
    no_observation = check_seq_explained(seqc)  # no observation in this particular time slice
    if no_observation:
        current_chunks_idx = []
        dt = int(1.0)
    else:
        while seq_explained == False:
            # find explanations for the upcoming sequence
            # record termination time of each biggest chunk used for explanation
            if PRINT:
                print('--------- before identifying one chunk ----------')
                print(' t is ', t)

            cg, seqc, chunk_record, explainchunk = identify_one_chunk(cg, seqc, explainchunk, chunk_record, t,
                                                                      candidate_set)
            seq_explained = check_seq_explained(seqc)
            if PRINT:
                print('--------- after identifying one chunk ----------')
        # chunkrecord: {time: [chunkindex]} stores the finishing time (exclusive) of each chunk
        # decide which are current chunks so that time is appropriately updated
        print(explainchunk)
        explainchunk.sort(key=lambda tup: tup[1])  # sort according to finishing time
        if len(seqc) >= 1: dt = min(min(explainchunk, key=lambda tup: tup[1])[1], seqc[0][0])
        else: dt = min(explainchunk, key=lambda tup: tup[1])[1]
        current_chunks_idx = [item[0] for item in explainchunk if item[1] == dt]  # move at the pace of the smallest identified chunk
        seq = seqc
        for chunk in current_chunks_idx:
            if chunk in cg.chunks:
                print('cg.chunks[chunk].T', cg.chunks[chunk].T)
    return current_chunks_idx, cg, dt, seq, chunk_record


def refactor(seq, current_chunk_idx, cg):
    # first, remove the identified current chunks from the sequence of observations
    if len(current_chunk_idx) > 0:
        for idx, _ in current_chunk_idx:
            content = cg.chunks[idx].content
            for tup in content:  # remove chunk content one by one from the sequence
                seq.remove(tup)

    seqcopy = []
    if seq != []:
        mintime = seq[0][0]
        for item in seq:
            listobs = list(item)
            listobs[0] = int(listobs[0] - mintime)
            # if listobs[0]>=0:
            seqcopy.append(
                tuple(listobs))  # there should not be negative ts, otherwise something is not explained properly
        return seqcopy
    else:
        return []


def check_subchunk_match(sequence_snapshot, chunk):
    """ check whether the subchunk of chunk matches with the sequence snapshot
        :returns True if there is a chunk match, false otherwise, when size is specified, the size of chunk match
        size: returns the size of the chunk, which is the number of observations that the chunk explains
        and also whether the sequence snapshot agrees with the chunk"""
    # whether the chunk could explain a subpart of the sequence snapshot
    elen = len(sequence_snapshot.intersection(chunk.content))
    return elen


def find_atomic_chunk(observation_to_explain):
    """:returns the first nonzero atomic unit"""

    H, W = observation_to_explain.shape[1:]
    for i in range(0, H):
        for j in range(0, W):
            chunk = np.zeros([1, H, W])
            if observation_to_explain[0, i, j] > 0:
                chunk[0, i, j] = observation_to_explain[0, i, j]  # group them by
                return chunk
    return np.zeros([1, H, W])


def rational_chunking(prev, current, combined_chunk, cg):
    """Updates chunking based on whether the new representation improves the evaluated loss function
        Updates the loss function associated with the new chunk, because the update of other chunks does not affect
        the loss function value estimation at the moment
        TODO: check the validity of this calculation method, possibly probability needs to be readjusted """
    w = 0.5  # tradeoff between accuracy and reaction time.
    cat = combined_chunk
    chunked = False
    P_s_giv_s_last = cg.get_transitional_p(prev, current)
    if P_s_giv_s_last is not None:
        Expected_reward_cat = P_s_giv_s_last * len(current) + (1 - P_s_giv_s_last) * (-1) * w * len(current)
        if Expected_reward_cat > 0:
            chunked = True
            cg.chunking_reorganization(prev, current, cat)
    return cg, chunked


def arr_to_tuple(arr):
    test_list = arr.tolist()
    res = tuple([tuple([tuple(ele) for ele in sub]) for sub in test_list])
    return res


def tuple_to_arr(tup):
    return np.array(tup)


def update_current_chunks(current_chunks, M):
    # update the marginal frequency of the current chunk
    if len(current_chunk) > 0:
        for chunk in current_chunks:
            if arr_to_tuple(chunk) not in list(M.keys()):
                M[arr_to_tuple(chunk)] = 1.0
            else:
                M[arr_to_tuple(chunk)] = M[arr_to_tuple(chunk)] + 1.0
    return M


def refactor_observation(observation, return_max_k=False):
    ''' observation: matrix to refactor
    max_k: the steps of reduction does the observation reduces.'''
    max_k = 0
    sum_obs = np.sum(np.abs(observation), axis=(1, 2))

    # TODO: improve the code of refactorization
    if observation.shape[0] > 1:
        for k in range(1, observation.shape[0]):
            # if np.sum(np.abs(observation[0:k, :, :])) == 0:
            if np.sum(sum_obs[0:k]) == 0:
                if k > max_k:
                    max_k = k
    observation = observation[max_k:, :, :]
    if return_max_k:
        return observation, max_k
    else:
        return observation


def check_chunk_ending(chunk, observation):
    '''Given that a chunk could explain the observation, check on whether it is ending at the last column of the observation'''
    if len(chunk) == len(observation):
        return True
    else:
        return False


# convert frequency into probabilities
def transition_into_probabilities(chunk_f, chunk_pair_f, prev, current):
    # decide if the number of exposure is significant enough to be concatinated together as a chunk
    if (prev in list(chunk_f.keys()) and prev in list(
            chunk_pair_f.keys())) and current in list(
        chunk_pair_f[prev].keys()):
        sum_transition = 0
        for key in list(chunk_pair_f[prev].keys()):
            sum_transition += chunk_pair_f[prev][key]
        sum_marginals = 0
        for key in list(chunk_f.keys()):
            sum_marginals = sum_marginals + chunk_f[key]
        P_prev = chunk_f[prev] / sum_marginals
        P_current_giv_prev = chunk_pair_f[prev][current] / sum_transition
    else:
        P_prev = None
        P_current_giv_prev = None
    return P_prev, P_current_giv_prev


def measure_average_EUEP(chunk_f):
    avg_euep = 0
    for key in list(chunk_f.keys()):
        probs = chunk_f[key] / sum(list(chunk_f.values()))
        avg_euep = avg_euep + probs * len(list(key)) / np.log2(probs)
    return avg_euep


from time import time


def convert_sequence(seq):
    Observations = []
    T, H, W = seq.shape
    for t in range(0, T):
        for h in range(0, H):
            for w in range(0, W):
                v = seq[t, h, w]
                if v != 0:
                    Observations.append((t, h, w, int(seq[t, h, w])))
    return Observations, T


import time


def save_diagnostic_data(data, parsingtime, learningtime, cg, current_chunks_idx, maxchunksize):
    data['parsing time'].append(parsingtime)
    data['learning time'].append(learningtime)
    data['n_chunk'].append(len(cg.chunks))
    if len(current_chunks_idx) > 0:
        sizes = []
        for ck in list(cg.chunks.values()):
            sz = ck.volume
            sizes.append(sz)
            if sz > maxchunksize: maxchunksize = sz
        data['chunk size'].append(maxchunksize)
    else:
        data['chunk size'].append(0)
    np.save('performance_data.npy', data)
    return data, maxchunksize


def save_chunk_record(chunkrecord, cg):
    import pickle
    ''' save chunk record for HCM learned on behaviorial data '''
    df = {}
    df['time'] = []
    df['chunksize'] = []
    for time in list(chunkrecord.keys()):
        df['time'].append(int(time))
        ckidx = chunkrecord[time][0][0]
        df['chunksize'].append(cg.chunks[ckidx].volume)
    with open('HCM_time_chunksize.pkl', 'wb') as f:
        pickle.dump(df, f)
    return


def hcm_learning(arayseq, cg, learn=True, abstraction=True):
    '''Sequence is an n-d array
        Note that sequence should not be too long, otherwise subtraction would take a while for the algorithm
        when the learning handle is false, then only parsing, and no chunking will take place '''
    seql, H, W = arayseq.shape
    cg.update_hw(H, W)
    seq_over = False
    chunk_record = {}  # chunk ending time, and which chunk is it.
    # dt: the time from the end of the previous chunk to the next chunk
    data = {'parsing time': [], 'learning time': [], 'n_chunk': [], 'chunk size': []}
    maxchunksize = 0
    if len(cg.chunks) > 0:
        sizes = []
        for ck in list(cg.chunks.values()):
            sz = ck.volume
            sizes.append(sz)
            if sz > maxchunksize: maxchunksize = sz

    seq, seql = convert_sequence(arayseq[0:maxchunksize + 1, :, :])  # loaded with the 0th observation
    print('INSTRUCTION SEQUENCE IS: ', seq)
    t = 0
    Buffer = buffer(t, seq, seql, arayseq.shape[0])

    while not seq_over:
        candidate_set = set(cg.chunks.values())  # print(cg, seq, chunk_record, Buffer.t, candidate_set)
        current_chunks_idx, cg, dt, seq, chunk_record = identify_latest_chunks(cg, seq, chunk_record, Buffer.t,
                                                                               candidate_set)  # identify latest ending chunks
        seq = Buffer.refactor(seq, dt)

        if len(current_chunks_idx) > 0 and learn == True:
            cg = learning_and_update(current_chunks_idx, chunk_record, cg, Buffer.t, threshold_chunk=True)
            if abstraction:
                cg = abstraction_update(current_chunks_idx, chunk_record, cg, Buffer.t, freq_T = 3)
        data, maxchunksize = save_diagnostic_data(data, -1, -1, cg, current_chunks_idx, maxchunksize)
        if learn == True: cg.forget()
        Buffer.reloadsize = maxchunksize + 1
        Buffer.checkreload(arayseq)
        seq_over = Buffer.checkseqover()
    return cg, chunk_record


def reload(seq, seql, arraysequence, t, max_chunksize):
    # reload arraysequence starting from time point t to the set sequence representations
    # t: the current time point
    T = seql  # current buffer length
    time = t + T
    relevantarray = arraysequence[time:time + max_chunksize, :, :]
    _, H, W = arraysequence.shape
    for tt in range(0, min(max_chunksize, relevantarray.shape[0])):
        for h in range(0, H):
            for w in range(0, W):
                v = relevantarray[tt, h, w]
                seq.append((T + tt, h, w, v))
    seql = seql + min(max_chunksize, relevantarray.shape[0])
    return seq, seql


def independence_test(pM, pT, N):
    # f_obs, f_exp = None, ddof = 0
    f_obs = []
    f_exp = []
    B = list(pM.keys())
    for cl in B:
        for cr in B:
            pclcr = pM[cl] * pM[cr] * N  # expected number of observations
            oclcr = pM[cl] * pT[cl][cr] * N  # number of observations
            f_exp.append(pclcr)
            f_obs.append(oclcr)
    df = (len(B) - 1) ** 2
    _, pvalue = stats.chisquare(f_obs, f_exp=f_exp, ddof=df)

    if pvalue < 0.05:
        return False  # reject independence hypothesis, there is a correlation
    else:
        return True


def get_minimal_complete_cks(seq, cg):
    for i in range(0, seq.shape[0]):
        atomic_chunk = np.zeros([1, 1, 1])
        atomic_chunk[0, 0, 0] = seq[i, 0, 0]
        if arr_to_tuple(atomic_chunk) not in list(cg.M.keys()):
            cg.M[arr_to_tuple(atomic_chunk)] = 1
            cg.T[arr_to_tuple(atomic_chunk)] = {}
            cg.add_chunk_to_vertex(arr_to_tuple(atomic_chunk))
        else:
            cg.M[arr_to_tuple(atomic_chunk)] = cg.M[arr_to_tuple(atomic_chunk)] + 1

    return cg


def rational_chunking_all_info(seq, cg, maxit=10):
    """" Rational chunk learner with the access to information about all the sequence
    Implemented in the hierarchical spatial temporal sequence
    """

    def hypothesis_test(cl, cr, pM, pT, N):
        p1p1 = pM[cl] * pM[cr]
        p1p0 = pM[cl] * (1 - pM[cr])
        p0p1 = (1 - pM[cl]) * pM[cr]
        p0p0 = (1 - pM[cl]) * (1 - pM[cr])

        op1p1 = pM[cl] * pT[cl][cr]
        op1p0 = pM[cl] * (1 - pT[cl][cr])
        op0p1 = 0
        op0p0 = 0

        for ncl in list(pT.keys()):
            if ncl != cl:
                op0p1 = op0p1 + pT[ncl][cr] * pM[ncl]
                op0p0 = op0p0 + (1 - pT[ncl][cr]) * pM[ncl]
        # if any cell contains less than 5 observations, need to
        if op0p0 * N <= 5 or op1p1 * N <= 5 or op1p0 * N <= 5 or op0p1 * N <= 5:
            return True  # cannot continute the test because of lack of sample size

        chisquare = N * (p1p1 * ((op1p1 - p1p1) / p1p1) ** 2 + p1p0 * ((op1p0 - p1p0) / p1p0) ** 2 + p0p1 * (
                    (op0p1 - p0p1) / p0p1) ** 2
                         + p0p0 * ((op0p0 - p0p0) / p0p0) ** 2)
        pvalue = stats.chi2.pdf(chisquare, 1)
        if pvalue < 0.0005:
            return False  # reject independence hypothesis, there is a correlation
        else:
            return True

    # get minimally complete chunk sets
    if len(cg.M) == 0:
        cg = get_minimal_complete_cks(seq, cg)

    if len(cg.M) > 0:
        pM, pT, _, N = partition_seq_hastily(seq, list(cg.M.keys()))

    proposed_chunk_p = {}
    for cl in list(pT.keys()):
        p_cl = pM[cl]
        for cr in list(pT[cl].keys()):
            cr_giv_cl = pT[cl][cr]
            p = p_cl * cr_giv_cl
            # check independence:
            if hypothesis_test(cl, cr, pM, pT, N) == False:  # there is a correlation
                if np.sum(tuple_to_arr(cl)) > 0 and np.sum(tuple_to_arr(cr)) > 0:
                    clcr = arr_to_tuple(np.concatenate((tuple_to_arr(cl), tuple_to_arr(cr)), axis=0))
                    # TODO: proposed chunk should rank with its parents.
                    proposed_chunk_p[clcr] = (cl, cr, p)
    sorted_proposal = {k: v for k, v in sorted(proposed_chunk_p.items(), key=lambda item: item[1][2])}

    # find the maximum transition probability, try chunking it
    it = 0  # number of iteration
    while independence_test(pM, pT, N) == False and len(list(sorted_proposal.keys())) > 0:
        # ''alternatively, while there are still dependencies between the chunks, keep searching for representation \n'
        # '        until they become independent')
        print(sorted_proposal.keys())
        new_chunk = list(sorted_proposal.keys())[-1]
        cl = sorted_proposal[new_chunk][0]
        cr = sorted_proposal[new_chunk][1]

        bag_of_chunks = list(pM.keys())
        while new_chunk in bag_of_chunks:
            print('the new chunk is already existing in the bag of chunks')
            sorted_proposal.pop(new_chunk)
            proposed_chunk_p.pop(new_chunk)
            new_chunk = list(sorted_proposal.keys())[-1]
            cl = sorted_proposal[new_chunk][0]
            cr = sorted_proposal[new_chunk][1]
        print(sorted_proposal)
        print(new_chunk)

        cg.add_chunk_to_vertex(new_chunk, left=cl, right=cr)
        new_bag_of_chunks = bag_of_chunks.copy()
        new_bag_of_chunks.append(new_chunk)
        new_pM, new_pT, _, N = partition_seq_hastily(seq, new_bag_of_chunks)
        pM = new_pM
        pT = new_pT
        cg.M = new_pM
        cg.T = new_pT

        proposed_chunk_p = {}
        for cl in list(new_pM.keys()):
            p_cl = new_pM[cl]
            for cr in list(new_pT[cl].keys()):
                cr_giv_cl = new_pT[cl][cr]
                p = p_cl * cr_giv_cl
                if hypothesis_test(cl, cr, pM, pT, N) == False:  # there is a correlation
                    if np.sum(tuple_to_arr(cl)) > 0 and np.sum(tuple_to_arr(cr)) > 0:
                        clcr = arr_to_tuple(np.concatenate((tuple_to_arr(cl), tuple_to_arr(cr)), axis=0))
                        proposed_chunk_p[clcr] = (cl, cr, p)
        sorted_proposal = {k: v for k, v in sorted(proposed_chunk_p.items(), key=lambda item: item[1][2])}

        it = it + 1
        if it > maxit or len(list(sorted_proposal.keys())) == 0:  # in case there is no proposal, exit.
            break

    return cg


# helper functions
def eval_pM_pT(bag_of_chunks, partitioned_sequence, freq=True):
    """checked"""
    M = {}
    T = {}
    for chunk in bag_of_chunks:
        parsed_chunk = chunk
        if freq == True:
            M[chunk] = get_pM_from_partitioned_sequence(parsed_chunk, partitioned_sequence, freq=True)
        else:
            M[chunk] = get_pM_from_partitioned_sequence(parsed_chunk, partitioned_sequence, freq=False)
    for cl in bag_of_chunks:
        T[cl] = {}
        for cr in bag_of_chunks:
            T[cl][cr] = get_pT_from_partitioned_sequence(cl, cr, partitioned_sequence, freq=freq)
    return M, T


def get_pT_from_partitioned_sequence(cl, cr, partitioned_sequence, freq=True):
    '''checked'''
    '''Get the estimated empirical probability of P(chunk2|chunk1),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0'''
    chunk1_count = 0
    chunk1_chunk2 = 0
    # the transition probability from chunk1 to chunk2
    # get P(chunk2|chunk1)
    not_over = True
    i = 0
    for candidate_chunk in partitioned_sequence:
        if candidate_chunk == cl:
            chunk1_count += 1
            if i + 1 < len(partitioned_sequence):
                candidate2 = partitioned_sequence[i + 1]
                if candidate2 == cr:
                    chunk1_chunk2 += 1
        i = i + 1
    if freq:
        return chunk1_count
    else:
        if chunk1_count > 0:
            return chunk1_chunk2 / chunk1_count
        else:
            return 0.0  # returns 0 if there is no occurrance for the first probabilility


def get_pM_from_partitioned_sequence(chunk, partitioned_sequence, freq=True):
    '''checked'''

    '''Get the estimated empirical probability of P(chunk),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0'''
    chunk1_count = 0
    for candidate_chunk in partitioned_sequence:
        if candidate_chunk == chunk:
            chunk1_count += 1
    if freq:
        return chunk1_count
    else:
        if chunk1_count > 0:
            return chunk1_count / len(partitioned_sequence)
        else:
            return 0.0
