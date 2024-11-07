def evaluate_KL_compared_to_ground_truth(reproduced_sequence, generative_marginals,cg):
    """compute conditional KL divergence between the reproduced sequence and the groundtruth"""
    """generative_marginal: marginals used in generating ground truth """
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    ground_truth_set_of_chunks = set(generative_marginals.keys())
    for chunk in generative_marginals.keys():
        cg.M[chunk] = 0

    learned_M, _, _,N = partition_seq_hastily(reproduced_sequence, list(cg.M.keys()))
    #learned_M = partition_seq_STC(reproduced_sequence, cg) # in this

    # compare the learned M with the generative marginals
    # Iterate over dictionary keys, and add key values to the np.array to be compared
    # based on the assumption that the generative marginals should have the same key as the probability ground truth.
    probability_learned = []
    probability_ground_truth = []
    for key in list(learned_M.keys()):
        probability_learned.append(learned_M[key])
        probability_ground_truth.append(generative_marginals[key])
    probability_learned = np.array(probability_learned)
    probability_ground_truth = np.array(probability_ground_truth)
    eps = 0.000000001
    EPS = np.ones(probability_ground_truth.shape)*eps
    # return divergence

    v_M1 = probability_ground_truth# input, q
    v_M1 = EPS + v_M1 # take out the protection to see what happens
    v_M2 = probability_learned # output p
    v_M2 = EPS + v_M2

    # calculate the kl divergence
    def kl_divergence(p, q):# usually, p is the output, and q is the input.
        return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p))) # KL divergence in units of bits.

    #p_log_p_div_q = np.multiply(v_M1,np.log(v_M1/v_M2)) # element wise multiplication
    KL = kl_divergence(v_M2,v_M1)
    # div = np.sum(np.matmul(v_M1.transpose(),p_log_p_div_q))
    return KL
