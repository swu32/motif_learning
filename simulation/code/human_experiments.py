

def c3_chunk_learning():
    def get_chunk_list(ck):
        #print(np.array(list(ck.content)))
        T = int(max(np.array(list(ck.content)).reshape([-1,4])[:, 0])+1)
        chunk = np.zeros([T],dtype=int)
        for t,_,_, v in ck.content:
            print(ck.content, chunk.size, T)
            chunk[t] = v
        for item in list(chunk):
            if item == 0:
                print('')
        return list(chunk)

    import pickle
    ''' save chunk record for HCM learned on behaviorial data '''
    df = {}
    df['time'] = []
    df['chunksize'] = []
    df['ID'] = []

    hcm_chunk_record = {}

    for ID in range(0, 50): # across 30 runs
        hcm_chunk_record[ID] = []
        seq = np.array(generateseq('c3', seql=600)).reshape((600, 1, 1))
        cg = CG1(DT=0.0, theta=0.92)  # initialize chunking part with specified parameters
        cg, chunkrecord = hcm_learning(seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        for time in list(chunkrecord.keys()):
            df['time'].append(int(time))
            ckidx = chunkrecord[time][0][0]
            df['chunksize'].append(cg.chunks[ckidx].volume)
            df['ID'].append(ID)
            chunk = get_chunk_list(cg.chunks[ckidx])
            hcm_chunk_record[ID].append(chunk)


    with open('HCM_time_chunksize.pkl', 'wb') as f:
        pickle.dump(df, f)

    with open('HCM_chunk.pkl', 'wb') as f:
        pickle.dump(hcm_chunk_record, f)

    return




def m1_m2():
    pass

    m1 = np.array([1,2,2,2, 2,2,1,1, 1,1,2,1]).reshape([-1, 1, 1])
    m2 = np.array([1,1,1,2, 2,1,1,2, 2,2,2,1]).reshape([-1, 1, 1])
    cgm1m2 = CG1(DT=0.1, theta=0.996)

    learned = False
    nrep = 0 # the number of repetition
    while ~learned:
        nrep = nrep + 1
        cgm1m2, chunkrecord = hcm_learning(m1, cgm1m2)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        if m1 in cgm1m2.chunks:
            learned = True


def simonsays():
    df = pd.read_csv('/Users/swu/Desktop/research/motif_learning/data/simonsays_ed/data.csv')

    dfm = {}  # model dataframe
    dfm['blockcollect'] = []
    dfm['ID'] = []
    dfm['condition'] = []
    dfm['correctcollect'] = []
    dfm['p'] = []
    dfm['trialcollect'] = []

    seql = 12
    len_train = 30
    len_test = 8
    def convert_sequence(seq):
        seq = list(seq)
        x = seq[0]
        proj_seq = [] # pause
        for item in seq:
            if item == x:
                proj_seq.append(1)
            else:
                proj_seq.append(2)
        return proj_seq
    def calculate_prob(chunk_record, cg):
        p = 1
        for key in list(chunk_record.keys()):# key is the encoding time
            p = p*cg.chunks[chunk_record[key][0][0]].count/np.sum([item.count for item in cg.chunks])
        return p

    for sub in np.unique(list(df['ID'])):
        # initialize chunking part with specified parameters
        cg = CG1(DT=0.1, theta=0.996)
        for trial in range(1, len_train + 3*len_test+ 1):
            ins_seq = df[(df['ID'] == sub)].iloc[(trial-1)*seql:trial*seql, :][
                'instructioncollect']
            condition = list(df[(df['ID'] == sub)].iloc[(trial-1)*seql:trial*seql, :][
                'condition'])[0]
            block = list(df[(df['ID'] == sub)].iloc[(trial-1)*seql:trial*seql, :][
                'blockcollect'])[0]
            proj_seq = convert_sequence(ins_seq)
            proj_seq = proj_seq  # display one time, and recall for another time
            proj_seq = np.array(proj_seq).reshape([-1, 1, 1])
            cg, chunkrecord = hcm_learning(proj_seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
            p_seq = calculate_prob(chunkrecord, cg)# evaluate the probability of a sequence
            recall_seq = cg.imagination1d(seql=12) # parse sequence using chunks and evaluate the chunk probability
            dfm['blockcollect'].append(block)
            dfm['ID'].append(sub)
            dfm['condition'].append(condition)
            dfm['correctcollect'].append(acc_eval1d(recall_seq, proj_seq))
            dfm['p'].append(p_seq)
            dfm['trialcollect'].append(trial)

    dfm = pd.DataFrame.from_dict(dfm)
    csv_save_directory = '/Users/swu/Desktop/research/motif_learning/data/simonsays/simulation_data_ed.csv'

    dfm.to_csv(csv_save_directory, index=False, header=True)

    return

def test_motif_learning_experiment2():
    training_seq, testing_seq = exp2()
    #training_seq, testing_seq = exp2(control = True)

    cg = CG1(DT=0.1, theta=0.996)
    for i in range(0, 40):
        proj_seq = training_seq[i]
        proj_seq = np.array(proj_seq).reshape([-1, 1, 1])
        cg, chunkrecord = hcm_learning(proj_seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        recalled_seq, ps = recall(cg, firstitem=proj_seq[0, 0, 0])
        if len(cg.variables.keys())>1:
            print()
        p_seq = np.prod(ps)  # evaluate the probability of a sequence
        print(p_seq)
    print('done with training')
    for i in range(0, 24):
        proj_seq = testing_seq[i]
        proj_seq = np.array(proj_seq).reshape([-1, 1, 1])
        cg, chunkrecord = hcm_learning(proj_seq, cg)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        recalled_seq, ps = recall(cg, firstitem=proj_seq[0, 0, 0])
        p_seq = np.prod(ps)  # evaluate the probability of a sequence
        print(recalled_seq, ps)

    return

