from Hand_made_generative import *
from Generative_Model import *
from text_learning import *
from Learning import *
from CG1 import *
from chunks import *
import numpy as np
import PIL as PIL
from PIL import Image
import os
from time import time
from chunks import *
from abstraction_test import *
#from simonsays import *

def measure_KL():
    '''Measurement of kl divergence across learning progress
    n_sample: number of samples used for a particular uncommital generative model
    d: depth of the generative model
    n: length of the sequence used to train the learning model'''
    df = {}
    df['N'] = []
    df['kl'] = []
    df['type'] = []
    df['d'] = []
    n_sample = 1  # eventually, take 100 runs to show such plots
    n_atomic = 5
    ds = [3, 4, 5, 6, 7, 8]
    Ns = np.arange(100,3000,100)
    for d in ds: # varying depth, and the corresponding generative model it makes
        depth = d
        for i in range(0, n_sample):
            # in every new sample, a generative model is proposed.
            cg_gt = generative_model_random_combination(D=depth, n=n_atomic)
            cg_gt = to_chunking_graph(cg_gt)
            for n in Ns:
                print({' d ': d, ' i ': i, ' n ': n })
                # cg_gt = hierarchy1d() #one dimensional chunks
                seq = generate_hierarchical_sequence(cg_gt.M, s_length=n)
                cg = Chunking_Graph(DT=0, theta=1)  # initialize chunking part with specified parameters
                cg = rational_chunking_all_info(seq, cg)
                imagined_seq = cg.imagination(n, sequential=True, spatial=False, spatial_temporal=False)
                kl = evaluate_KL_compared_to_ground_truth(imagined_seq, cg_gt.M, Chunking_Graph(DT=0, theta=1))

                # take in data:
                df['N'].append(n)
                df['d'].append(depth)
                df['kl'].append(kl)
                df['type'].append('ck')

    df = pd.DataFrame.from_dict(df)
    df.to_pickle('KL_rational_learning_N')  # where to save it, usually as a .pkl
    return df


def p_RNN(trainingseq,testingseq):
    # need a list of prediction and output probability.
    # train until the next mistake.
    '''Compare neural network behavior with human on chunk prediction'''
    sequence = np.array(trainingseq).reshape((-1,1,1))
    sequence[0:5,:,:] = np.array([0,1,2,3,4]).reshape((5,1,1))
    #sequence = np.array(generateseq('c3', seql=600)).reshape((600, 1, 1))
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--sequence-length', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    args = parser.parse_args()


    dataset = Dataset(sequence, args)  # use all of the past dataset to train
    model = Model(dataset)
    train(dataset, model, args)  # train another model from scratch.

    start = 10
    prob = [0.25]*start
    testsequence = np.array(testingseq).reshape((-1,1,1))
    for idx in range(start, testsequence.shape[0]):
        pre_l = 10
        p_next = evaluate_next_word_probability(model, testsequence[idx], words=list(testsequence[max(idx-pre_l,0):idx,:,:].flatten()))
        prob.append(p_next[0][0])

    return prob



def NN_data_record():
    ################# Training Neural Networks to Compare with Learning Sequence ###########

    df = {}

    df['N'] = []
    df['klnn'] = []
    n_sample = 5  # taking 10 samples for each of the N specifications.
    Ns = np.arange(50, 3000, 50)

    cg_gt = generative_model_random_combination(D=3, n=5)
    cg_gt = to_chunking_graph(cg_gt)

    for i in range(0, n_sample):
        # Ns = np.arange(100,3000,100)

        for j in range(0, len(Ns)):
            n = Ns[j]
            seq = generate_hierarchical_sequence(cg_gt.M, s_length=n)
            print(len(seq))
            imagined_seq = NN_testing(seq)
            imagined_seq = np.array(imagined_seq).reshape([len(imagined_seq),1,1])
            kl = evaluate_KL_compared_to_ground_truth(imagined_seq, cg_gt.M, Chunking_Graph(DT=0, theta=1))
            df['N'].append(n)
            df['klnn'].append(kl)
            print({'kl is ': kl})

    df = pd.DataFrame.from_dict(df)
    df.to_pickle('../KL_neural_network_N')  # where to save it, usually as a .pkl
    return


def evaluate_perplexity(data, chunkrecord):
    #TODO: convert chunkrecord into sequence of probability
    p = []
    n_ck = 0
    for t in range(0, len(data)):
        if t in list(chunkrecord.keys()):
            freq = chunkrecord[t][0][1]
            n_ck = n_ck + 1
            p.append(freq/n_ck)
        else: # a within-chunk element
            p.append(1)
    perplexity = 2**(-np.sum(np.log2(np.array(p)))/len(p))

    return perplexity



def plot_model_learning_comparison(cg1, cg2):
    import matplotlib.pyplot as plt
    import numpy as np

    titles = ['parsing length', 'representation complexity', 'explanatory volume', 'sequence complexity',
              'representation entropy', 'n chunks', 'n variables','storage cost']
    units = ['n chunk', 'bits', 'l', 'bits', 'bits', 'n chunk', 'n variable','bits']
    ld1 = np.array(cg1.learning_data)
    ld2 = np.array(cg2.learning_data)
    # Create a figure and subplots with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 4, figsize=(10, 6))
    x = np.cumsum(ld1[:, 0])

    for i, ax in enumerate(axs.flat):
        if i >= 8:
            break
        y1 = ld1[:, i + 1]
        y2 = ld2[:, i + 1]
        ax.plot(x, y1, label='HCM')
        ax.plot(x, y2, label='HVM')
        ax.set_title(titles[i])
        ax.set_ylabel(units[i])
        ax.set_xlabel('Sequence Length')
    # Adjust spacing between subplots
    fig.tight_layout()
    # Show the figure
    plt.legend()
    plt.show()
    # save the figure
    fig.savefig('modelcomparison.png')
    return


def plot_average_model_learning_comparison(datahcm, datahvm, d=None, sz=10, savename = 'modelcomparison.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    # both are three dimensional arrays

    titles = ['parsing length', 'representation complexity', 'explanatory volume', 'sequence complexity',
              'representation entropy', 'n chunks', 'n variables', 'storage cost']

    units = ['n chunk', 'bits', 'l', 'bits', 'bits', 'n chunk', 'n variable', 'bits']
    # Create a figure and subplots with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 4, figsize=(10, 6))
    x = np.cumsum(datahcm[0,:, 0])

    gt = np.load('./data/generative_hvm' + ' d = ' + str(d) + 'sz = ' + str(sz) + '.npy')


    for i, ax in enumerate(axs.flat):
        if i >= 8:
            break
        hcm_mean = np.mean(datahcm[:, :, i + 1], axis = 0)
        hvm_mean = np.mean(datahvm[:, :, i + 1], axis = 0)
        ax.plot(x, hcm_mean, label='HCM', color='orange', linewidth=4, alpha=0.3)
        ax.plot(x, hvm_mean, label='HVM', color='blue', linewidth=4, alpha=0.3)
        ax.plot(x, [gt[0,i + 1]]*len(x), label='GT', color='green', linewidth=4, alpha=0.3)
        for j in range(0, datahcm.shape[0]):
            ax.plot(x, datahcm[j, :, i + 1], color='orange', linewidth=1, alpha = 0.3)
            ax.plot(x, datahvm[j, :, i + 1], color='blue', linewidth=1, alpha = 0.3)

        ax.set_title(titles[i])
        ax.set_ylabel(units[i])
        ax.set_xlabel('Sequence Length')
    # Adjust spacing between subplots
    fig.tight_layout()
    # Show the figure
    plt.legend()
    plt.show()
    # save the figure
    fig.savefig(savename)

    return


def evaluate_random_graph_abstraction():
    '''Compare HCM and HVM on random abstract represnetation graph '''
    #cggt, seq = random_abstract_representation_graph(save=True)
    # with open('random_abstract_sequence.npy', 'rb') as f:
    #     seq = np.load(f)
    with open('sample_abstract_sequence.npy', 'rb') as f:
        seq = np.load(f)

    cghcm = CG1(DT=0.1, theta=0.996)
    cghcm = hcm_markov_control(seq, cghcm, ABS = False)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    cghvm = CG1(DT=0.1, theta=0.996)
    cghvm = hcm_markov_control(seq, cghvm)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    plot_model_learning_comparison(cghcm, cghvm)
    return


def test_random_graph_abstraction(generation = False):
    depth_increment = [30, 40, 50] # [5, 10, 15, 20, 25, 30, 35, 40]
    if generation:
        for d in depth_increment:
            random_abstract_representation_graph(save=True, alphabet_size=10, depth=d, seql = 5000)

    # with open('random_abstract_sequence.npy', 'rb') as f:
    #     seq = np.load(f)
    for sz in depth_increment:
        openpath = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = ' + str(sz) + '.npy'
        with open(openpath, 'rb') as f:
            fullseq = np.load(f)
        slice_sz = 5000
        n_measure = 9
        n_run = 5#int(len(fullseq)/slice_sz)
        n_iter = 10
        datahcm = np.empty((n_run, n_iter, n_measure))
        datahvm = np.empty((n_run, n_iter, n_measure))
        i = 0
        for seq in slicer(fullseq, slice_sz):
            cghvm = CG1(DT=0.1, theta=0.996)
            cghvm = hcm_markov_control(seq, cghvm, MAXit = n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

            cghcm = CG1(DT=0.1, theta=0.996)
            cghcm = hcm_markov_control(seq, cghcm, ABS=False, MAXit = n_iter)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

            datahcm[i,:,:] = np.array(cghcm.learning_data)
            datahvm[i,:,:] = np.array(cghvm.learning_data)
            if i == n_run: break # just do 1 iteration
            i = i + 1
        np.save('./data/hcm_fixed_support_set' + ' d = ' + str(sz) + '.npy', datahcm)
        np.save('./data/hvm_fixed_support_set' + ' d = ' + str(sz) + '.npy', datahvm)
        plot_average_model_learning_comparison(datahcm, datahvm, d = sz, savename = './data/fixed_support_set' + ' d = ' + str(sz) + '.png')
    return



def slicer(seq, size):
    """Divide the sequence into chunks of the given size."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



def test_depth_parsing():

    with open('sample_abstract_sequence.npy', 'rb') as f:
        seq = np.load(f)

    cghcm = CG1(DT=0.1, theta=0.996)
    cghcm = hcm_depth_parsing(seq, cghcm)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    return



def test_random_graph_abstraction_recursive_learning():
    # test the process of parsing the sequence with varying depth level
    with open('sample_abstract_sequence.npy', 'rb') as f:
        seq = np.load(f)

    cghcm = CG1(DT=0.1, theta=0.996)
    cghcm = hcm_depth_parsing(seq, cghcm)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    plot_model_learning_comparison(cghcm, cghvm)
    return

def test_simple_abstraction():
    seq = simple_abstraction_I()
    cg = CG1(DT=0.1, theta=0.996)
    cg, chunkrecord = hcm_rational(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)

    return


def main():
    test_random_graph_abstraction(generation=False)
    seq = abstraction_illustration()

    test_depth_parsing()
    simonsaysex2()
    test_simple_abstraction()# within which there is an hcm rational

    pass

if __name__ == "__main__":

    main()

