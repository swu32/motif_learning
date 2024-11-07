import numpy as np
import random
from Learning import *
from CG1 import *


def initialize(d,cg):
    ''' Add atomic chunks to chunking graph '''
    for i in range(1,d+1):
        newchunk = Chunk([(0,0,0,i)])
        cg.add_chunk(newchunk, ancestor=True)
    return cg

def timeshift(content, t):
    shiftedcontent = []
    for tup in list(content):
        lp = list(tup)
        lp[0] = lp[0] + t
        shiftedcontent.append(tuple(lp))
    return set(shiftedcontent)


def connect_chunks(chunklist):
    ''' connect chunk one after another, and add them to the current chunking graph '''
    combined_chunk_content = []
    variables = {}
    for ck in chunklist:
        combined_chunk_content = combined_chunk_content + ck.ordered_content
        if type(ck)== Variable:
            variables = variables | {ck.key: ck}
        else:
            variables = variables | ck.includedvariables

    newchunk = Chunk(([]), includedvariables = variables, ordered_content = list(combined_chunk_content))
    return newchunk


def random_abstract_representation_graph(save = True, alphabet_size = 10, depth = 5, seql = 1000):
    ''' Generate a random abstract representation graph '''
    n_chunk_combo_range = [2,3,4,5]
    n_variable_entailment_range = [2,3,4,5]
    n_sample = 8000
    cg = CG1()
    cg = initialize(alphabet_size, cg)
    seql = seql # length of the sequence subsegment


    for _ in range(depth):
        RAND = np.random.rand()# create a chunk or a variable with 50% probability
        if RAND > 0.5:# create chunks
            B = list(cg.chunks.values()) + list(cg.variables.values()) #belief set.
            n_combo = np.random.choice(n_chunk_combo_range)
            samples = random.choices(B, k = n_combo)
            while type(samples[0])!=Chunk or type(samples[-1])!=Chunk:
                samples = random.sample(B, k=n_combo)
            print(' combine existing chunks ')
            for item in samples:
                print(item.key)
            newchunk = connect_chunks(samples)
            cg.add_chunk(newchunk)
        else: # create variables
            B = list(cg.chunks.values())  # belief set
            n_combo = np.random.choice(n_variable_entailment_range)
            samples = random.sample(B, k=n_combo)
            sampledict = dict(zip([item.key for item in samples], samples))
            newvariable = Variable(sampledict)
            cg.add_variable(newvariable,sampledict)
            print('make new variables')
            print([item.key for item in samples])

            #cg.add_variable(newvariable, set([item.key for item in samples]))

    cg = assign_probabilities(cg)
    sampled_seq = cg.sample_chunk(60000)
    seq, total_length = convert_chunklist_to_seq(sampled_seq, cg, seql=30000)

    #lz_complexity, lz_seql = lzcompression(seq[:seql, :, :])

    # record ground truth metric on complexities and others

    # sc: average sequence complexity per sequence length of 1000
    sc = sum([-np.log2(cg.chunk_probabilities[name]) for name in sampled_seq])/total_length*seql # calibrated to the sequence length of 1000
    # re: representation entropy per sequence length of 1000
    re = cg.calculate_representation_entropy(sampled_seq, gt=True)/total_length*seql
    # stc: storage costrandom_abstract_representation_graph
    stc = cg.calculate_storage_cost(gt=True)
    nc, nv = len(cg.chunks), len(cg.variables)
    cg.learning_data.append([total_length, float("nan"), cg.calculate_rc(), total_length / len(sampled_seq), sc, re, nc, nv, stc])
    print('generative model ==================================',)
    print('parsing length', seql/(total_length/len(sampled_seq)), 'sequence complexity', sc, 'representation complexity ',  cg.calculate_rc(), ' explanatory volume ', total_length / len(sampled_seq),
     'representation entropy', re, 'n chunks', nc, 'n variables', nv, 'storage cost', stc)
    #print('lz  =====================================',)
    #print('lz_seql ', lz_seql, ' lz_complexity ', lz_complexity)
    np.save('./data/generative_hvm' + ' d = ' + str(depth) + 'sz = ' + str(alphabet_size) + '.npy', np.array(cg.learning_data))

    if save:
        savename = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = ' + str(depth) + '.npy'
        with open(savename, 'wb') as f:
            np.save(f, seq)

    return cg, seq

def convert_chunklist_to_seq(sampled_seq, cg, sparse = True, seql = 1000):
    '''convert a sequence of sampled chunk to a sequence of observations'''

    content_list = []
    for key in sampled_seq:
        chunk = cg.chunks[key]
        cg.sample_variable_instances() # resample variables in cg again
        full_content = cg.get_concrete_content(chunk)
        content_list.append(full_content)
    seq = np.zeros([seql,1,1])
    t = 0
    if sparse: # insert empty spaces between chunks
        for orderedcontent in content_list:# each is an ordered content
            for chunk in orderedcontent:# each chunk is a set
                for element in chunk:
                    try:
                        if t + element[0]<seql:
                            seq[t+element[0], element[1], element[2]] = element[3]
                    except(TypeError):
                        print()
                t = t + int(max(np.atleast_2d(np.array(list(chunk)))[:, 0]) + 1)

    return seq, t

def assign_probabilities(cg):
    """Assign random probablities for independent chunks in cg in addition to variables within"""
    ps = dirichlet_flat(len(cg.chunks), sort=False)
    cg.chunk_probabilities = dict(zip(list(cg.chunks.keys()), ps))
    for var in cg.variables.values():
        ps = dirichlet_flat(len(set(var.entailingchunks)), sort=False)
        var.chunk_probabilities = dict(zip(list(var.entailingchunks), ps))
    return cg


''''Generates a hierarchical generative model with depth d'''
def generate_hierarchical_sequence(marginals, s_length=1000):
    # marginals can allso be the learned marginals, in that case this function is used to produce a simulated sequence
    # spatial or temporal
    # spatial chunks: chunks that exist only in spatial domain
    # temporal chunks: chunks that exist only in temporal domain
    # spatial temporal chunks: chunks that exist both in spatial and temporal domain.
    # or temporal sequential chunks

    not_over = True
    while not_over:
        new_sample, _ = sample_from_distribution(list(marginals.keys()), list(marginals.values()))
        # what is this sample from distribution when the marginals contain spatial temporal chunks?
        sequence = sequence + new_sample
        if len(sequence) >= s_length:
            not_over = False
            sequence = sequence[0:s_length]
    return sequence

# seq= generate_hierarchical_sequence(generative_marginals,s_length = 1000)

def sample_from_distribution(states, prob):
    """
    states: a list of states to sample from
    prob: another list that contains the probability"""
    prob = [k / sum(prob) for k in prob]
    cdf = [0.0]
    for s in range(0, len(states)):
        cdf.append(cdf[s] + prob[s])
    k = np.random.rand()
    for i in range(1, len(states) + 1):
        if (k >= cdf[i - 1]):
            if (k < cdf[i]):
                return list(states[i - 1]), prob[i - 1]


def partition_seq(this_sequence, bag_of_chunks):
    '''one dimensional chunks, multi dimensional chunks, TODO: make them compatible to other chunks'''
    # find the maximal chunk that fits the sequence
    # so that this could be used to evaluate the learning ability of the algorithm
    # what to do when the bag of chunks does not partition the sequence??
    i = 0
    end_of_sequence = False
    partitioned_sequence = []

    while end_of_sequence == False:
        max_chunk = None
        max_length = 0
        for chunk in bag_of_chunks:
            this_chunk = json.loads(chunk)
            if this_sequence[i:i + len(this_chunk)] == this_chunk:
                if len(this_chunk) > max_length:
                    max_chunk = this_chunk
                    max_length = len(max_chunk)

        if max_chunk == None:
            partitioned_sequence.append([this_sequence[i]])
            i = i + 1
        else:
            partitioned_sequence.append(list(max_chunk))
            i = i + len(max_chunk)

        if i >= len(this_sequence): end_of_sequence = True

    return partitioned_sequence

def dirichlet_flat(N, sort = True):
    alpha = tuple([1 for i in range(0, N)])  # coefficient for the flat dirichlet distribution
    if sort:return sorted(list(np.random.dirichlet(alpha, 1)[0]), reverse=True)
    else: return list(np.random.dirichlet(alpha, 1)[0])

import itertools


def generate_new_chunk(setofchunks):
    zero = arr_to_tuple(np.zeros([1,1,1]))
    a = list(setofchunks)[
        np.random.choice(np.arange(0, len(setofchunks), 1))]  # better to be to choose based on occurrance probability
    b = list(setofchunks)[np.random.choice(np.arange(0, len(setofchunks), 1))]  # should exclude 0
    va, vb = tuple_to_arr(a), tuple_to_arr(b)
    la, lb = va.shape[0], vb.shape[0]
    lab = la + lb
    vab = np.zeros([lab, 1, 1])
    vab[0:va.shape[0], :, :] = va
    vab[va.shape[0]:, :, :] = vb
    ab = arr_to_tuple(vab)
    print('ab = ', ab,' a = ', a,' b = ',b)
    if ab in setofchunks or np.array_equal(a, zero) or np.array_equal(b, zero):
        generate_new_chunk(setofchunks)
    else:
        return ab, a, b


def overlap_but_different_graph(cgg, n_s, n_d, n_atom):
    """ n_s: the number of shared chunks other than the atomic chunks
        n_d: the number of different chunks
        n_atom: the number of atomic chunks in cg"""

    def check_independence(constraints, M):
        for ab, a, b in constraints:
            if M[ab] <= M[a] * M[b] + 0.003:
                return False  # constraint is not satisified
        return True

    not_allowed_chunks = list(cgg.M.keys())
    # trim the graph until the overlapping number of nodes are allowed to stay
    cg = trim_graph(cgg, len(list(cgg.M.keys())) - n_atom - n_s) # cg is another graph instance

    setofchunks = list(cg.M.keys())
    setofchunkswithoutzero = setofchunks.copy()
    setofchunkswithoutzero.remove(arr_to_tuple(np.zeros([1,1,1])))
    constraints =  []
    for d in range(0, n_d):# iteratively create different chunks from the generative chunk grpah
        # pick random, new combinations
        ab, a, b = generate_new_chunk(setofchunkswithoutzero)
        while ab in setofchunks or ab in not_allowed_chunks:
            ab, a, b = generate_new_chunk(setofchunkswithoutzero)
        constraints.append([ab,a,b])
        setofchunks.append(ab)

    # calculate the chunk occurance probabilities, if the chunks are combined independently, given this
    # distribution assignment.
    satisfy_constraint = False
    while satisfy_constraint==False:
        # assign probabilities to this set of chunks:
        genp = dirichlet_flat(len(setofchunks), sort=False)
        p0 = max(genp)
        genp.remove(p0)
        p = [p0] + genp
        # normalize again, sometimes they don't sum up to 1.
        M = dict(zip(setofchunks, p))  # so that empty observation is always ranked the highest.
        cg.M = M
        satisfy_constraint = check_independence(constraints, M)

    return cg





def generative_model_random_combination(D=6, n=5):
    """ randomly generate a set of hierarchical chunks
        D: number of recombinations
        n: number of atomic, elemetary chunks"""
    def check_independence(constraints, M):
        for ab, a, b in constraints:
            if M[ab]<=M[a]*M[b]+ 0.003:
                return False # constraint is not satisified
        return True

    cg = Chunking_Graph()
    setofchunks = []
    for i in range(0, n):
        zero = np.zeros([1,1,1])
        zero[0,0,0] = i
        chunk = zero
        setofchunks.append(arr_to_tuple(chunk))

    setofchunkswithoutzero = setofchunks.copy()
    setofchunkswithoutzero.remove(arr_to_tuple(np.zeros([1,1,1])))
    constraints =  []
    for d in range(0, D):
        # pick random, new combinations
        ab, a, b = generate_new_chunk(setofchunkswithoutzero)
        # while ab in setofchunks:# keep generating new chunks that is new
        #     ab, a, b = generate_new_chunk(setofchunkswithoutzero)
        constraints.append([ab,a,b])
        setofchunks.append(ab)

    # calculate the chunk occurance probabilities, if the chunks are combined independently, given this
    # distribution assignment.
    satisfy_constraint = False
    while satisfy_constraint==False:
        # assign probabilities to this set of chunks:
        genp = dirichlet_flat(len(setofchunks), sort=False)
        p0 = max(genp)
        genp.remove(p0)
        p = [p0] + genp
        # normalize again, sometimes they don't sum up to 1.
        M = dict(zip(setofchunks, p))  # so that empty observation is always ranked the highest.
        cg.M = M
        satisfy_constraint = check_independence(constraints, M)

    # joint chunk probabilties needs to be higher than the independence criteria, otherwise items are not going to be
    # chunked.
    return cg


def to_chunking_graph(cg):
    M = cg.M

    # first, filter out the best joint probability from the marginals
    # then find out
    atomic_chunks = find_atomic_chunks(M)# initialize with atomic chunks
    for ac in atomic_chunks:
        cg.add_chunk_to_vertex(ac)

    chunks = set()
    for ck in list(atomic_chunks.keys()):
        chunks.add(ck)

    print(chunks)
    complete = False
    proposed_joints = set()
    while complete == False:
        # calculate the mother chunk and the father chunk of the joint chunks
        joints_and_freq = calculate_joints(chunks, M)# the expected number of joint observations
        new_chunk, cl, cr = pick_chunk_with_max_prob(joints_and_freq)
        while new_chunk in proposed_joints:
            joints_and_freq.pop(new_chunk)
            new_chunk, cl, cr = pick_chunk_with_max_prob(joints_and_freq)

        cg.add_chunk_to_vertex(new_chunk, left=cl, right=cr) #update cg graph with newchunk and its components
        chunks.add(new_chunk)
        proposed_joints.add(new_chunk)
        complete = check_completeness(chunks,M)
    return cg





def get_n(chunk,Mchunk):
    mck = tuple_to_arr(Mchunk)
    ck = tuple_to_arr(chunk)
    n = 0
    for i in range(0, mck.shape[0]):
        # find the maximally fitting chunk in bag of chunks to partition Mchunk
        if np.array_equal(mck[i:min(i+ck.shape[0],mck.shape[0]),:,:],ck):
            n = n + 1
    return n


def calculate_joints(chunks, M):
    """Calculate the best joints to combine with the pre-existing chunks"""
    ZERO = arr_to_tuple(np.zeros([1,1,1]))
    joints_and_freq = {}
    for chunk1 in chunks:
        for chunk2 in chunks:
            if chunk1 != ZERO and chunk2 != ZERO:
                chunk12 = combine_joints(chunk1, chunk2)
                #augmented_chunks = chunks.add(chunk12)
                # now use augmented chunks to evaluate the probability given the generative model M
                #chunk_dict = calculated_new_chunk_probability(chunks, M)
                ck_prob = calculate_expected_occurrance(chunk12,chunks,M) # need to use the entire chunk set to
                # calculate
                # expected occurrance.
                joints_and_freq[chunk12] = (ck_prob[chunk12],chunk1,chunk2)
    return joints_and_freq

def calculate_new_chunk_prob(chunks, M):
    """Given a set of bag of chunks, usually containing one that is a proposed new chunk, calculate hte
    occurrance probability"""
    # calculate frequency of count of each of the chunk in M, with size precendency
    # calculate normalized
    pass
def combine_joints(chunk1,chunk2):
    c1 = tuple_to_arr(chunk1)
    c2 = tuple_to_arr(chunk2)
    chunk12 = np.zeros([c1.shape[0] + c2.shape[0], 1, 1])
    chunk12[0:c1.shape[0], :, :] = c1
    chunk12[c1.shape[0]:, :, :] = c2
    chunk12 = arr_to_tuple(chunk12)
    return chunk12


def calculate_expected_occurrance(chunk12,bagofchunks,M):
    """In case when [1] [2] and [1,2] both exist, evaluation of [1] and [2] is included in [1,2]
     """
    # alternatively, partition M according to the new chunks.
    newchunks = bagofchunks.union({chunk12})# new bag of chunks when chunk12 is appended.
    ckfreq = {}
    for chunk in newchunks:
        ckfreq[chunk] = 0


    for gck in list(M.keys()):
        gcka = tuple_to_arr(gck)
        lgck = gcka.shape[0]

        ckupdate = {}
        for chunk in newchunks:
            ckupdate[chunk] = 0

        l =0

        while l<lgck:
            maxl = 0
            best_fit = None
            for cuk in newchunks:
                ck = tuple_to_arr(cuk)
                lck = ck.shape[0]
                if np.array_equal(gcka[l:min(l + lck, lgck), :, :], ck) and lck >= maxl:
                    best_fit = cuk
                    maxl = lck
            l = l + maxl
            ckupdate[best_fit] = ckupdate[best_fit] + 1

        for chunk in list(ckupdate.keys()):
            ckupdate[chunk] = ckupdate[chunk] * M[gck]
            ckfreq[chunk] = ckfreq[chunk] + ckupdate[chunk]

    # normalization
    SUM =np.sum(list(ckfreq.values()))
    for chunk in list(ckfreq.keys()):
        ckfreq[chunk] = ckfreq[chunk]/SUM

    return ckfreq

