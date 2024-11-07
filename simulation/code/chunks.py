import random
import string
import numpy as np


class Chunk:
    def __init__(self, chunkcontent, includedvariables={}, ordered_content=None, count=0, H=1, W=1, pad=1):
        """chunkcontent: a list of tuples describing the location and the value of observation
            includedvariables: a dictionary of variables that are included in this chunk
            ordered_content: a list of sets, each set contains the content of a chunk
            count: the number of times this chunk has been observed
            H: height of the chunk
            W: width of the chunk
            pad: boundary size for nonadjacency detection, set the pad to not 1 to enable this feature.
            """
        # TODO: make sure that there is no redundency variable
        ########################### Content and Property ########################
        #self.current_chunk_content() # dynamic value, to become the real content for variable representations
        if ordered_content!=None:
            self.ordered_content = ordered_content
            self.key = ''
            for item in ordered_content:
                eachcontent = item
                if type(eachcontent) == str:
                    self.key = self.key + eachcontent
                else:
                    self.key = self.key + str(tuple(sorted(eachcontent)))
        else:
            try:
                self.ordered_content = [set(chunkcontent)] #specify the order of chunks and variables
                self.key = tuple(sorted(chunkcontent))
            except(TypeError):
                print()
        if len(includedvariables)==0: self.includedvariables = {}
        else:
            self.includedvariables = includedvariables
            try:
                for key, var in self.includedvariables.items():
                    var.chunks[self.key] = self
            except(AttributeError):
                print('set object')
        try:
            self.content = self.get_content(self.ordered_content)
        except(AttributeError):
            print('')

        self.volume = sum([len(chunkcontent) for chunkcontent in self.ordered_content])# number of distinct item in the chunk (variable count as one item)
        self.T = 0 if chunkcontent == list([]) else int(max(np.atleast_2d(np.array(list(self.content)))[:, 0]) + 1)# number of time steps spanned by the chunk
        self.H = H
        self.W = W
        self.vertex_location = None
        self.pad = pad  # boundary size for nonadjacency detection, set the pad to not 1 to enable this feature.
        self.count = count  #
        self.birth = None  # chunk creation time
        self.entropy = 0

        ########################### Relational Connection ########################
        self.adjacency = {} # chunk --> something
        self.preadjacency = {} # something --> chunk
        self.indexloc = self.get_index() # TODO: index location
        self.arraycontent = None
        self.boundarycontent = set()

        self.abstraction = {}  # the variables summarizing this chunk
        self.all_abstraction = set() # all of the chunk's abstractions
        self.cl = {}  # left decendent
        self.cr = {}  # right decendent
        self.acl = {} # left ancestor
        self.acr = {} # right ancestor

        ###################### discount coefficient for similarity computation ########################
        # T, H, W, cRidx = self.get_index_padded() # TODO: index location
        self.D = 10
        self.matching_threshold = 0.8
        self.matching_seq = {}
        self.h = 1.
        self.w = 1.
        self.v = 1.
        self.parse = 0

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __ne__(self, other):
        return not(self == other)


    def get_random_name(self):
        length = 4
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str


    def get_content(self,ignore_variable = True):
        # returns a set with content and their specified locations
        if len(self.ordered_content)==1:
            return self.ordered_content[0]
        else:
            if ignore_variable:
                return
            else:
                # return the variable instantiated content
                content_set = set()
                tshift = 0
                for content in self.ordered_content:
                    maxdt = 0
                    for signal in content:
                        if signal[0] >= maxdt:
                            maxdt = signal[0]

                        tshiftsignal = list(signal).copy()
                        tshiftsignal[0] = tshiftsignal[0] + tshift
                        content_set.add(tuple(tshiftsignal))

                    tshift = tshift + maxdt + 1
                return content_set


    def get_full_content(self):
        '''returns a list of all possible content that this chunk can take'''
        self.possible_path = []
        self.get_content_recursive(self, [])
        return self.possible_path

    def get_content_recursive(self, node, path):
        '''This function does not fully work'''
        if node.content!=None:
            path = path + list(node.content)
        if len(list(node.includedvariables)) == 0:
            self.possible_path.append(path)
            return
        else:
            for Var in node.includedvariables:
                self.get_content_recursive(Var, path)


    def update_variable_count(self):
        for var in list(self.abstraction.values()):
            var.update()
        return

    def update(self):
        self.count = self.count + 1
        # update the count of the abstracted variables
        if len(self.abstraction) > 0:
            self.update_variable_count()

        # update the identification frequency of the included variables
        for k, v in self.includedvariables.items():
            v.identificationfreq += 1

        return

    def to_array(self):
        '''convert the content into array'''
        # TODO: correct self.T throughout the program
        arrep = np.zeros((int(max(np.atleast_2d(np.array(list(self.content)))[:, 0]) + 1), self.H, self.W))
        for t, i, j, v in self.content:
            arrep[t, i, j] = v
        self.arraycontent = arrep
        return

    def get_N_transition(self, dt):
        N = 0
        for chunk in self.adjacency:
            if dt in list(self.adjacency[chunk].keys()):
                N = N + self.adjacency[chunk][dt]
        return N

    def get_index(self):
        ''' Get index location of the concrete chunks in chunk content, variable index is not yet integrated '''
        if len(self.ordered_content)==0:
            return set()
        elif len(self.includedvariables) >0:
            return set() # give up when there are variables in the sequence
        else:
            # TODO: integrate with ordered chunkcontent
            index_set = set()

            index0 = set(map(tuple, np.atleast_2d(list(self.ordered_content[0]))[:, 0:3]))
            try:
                t_shift = int(np.atleast_2d(list(self.ordered_content[0]))[:, 0].max() + 1)  # time shift is the biggest value in the 0th dimension
            except(TypeError):
                print('')
            index_set.update(index0)
            for item in self.ordered_content[1:]:
                if type(item)!= str:
                    index = set(map(tuple, np.atleast_2d(list(item))[:, 0:3]))
                    shifted_index = self.timeshift(index, t_shift)
                    index_set.update(shifted_index)
                    t_shift = int(np.atleast_2d(list(item))[:, 0].max() + 1)# time shift is the biggest value in the 0th dimension

            return index_set

    def timeshift(self, content, t):
        shiftedcontent = []
        for tup in list(content):
            lp = list(tup)
            lp[0] = lp[0] + t
            shiftedcontent.append(tuple(lp))
        return set(shiftedcontent)

    def get_index_padded(self):
        # TODO: integrate with ordered chunkcontent

        ''' Get padded index arund the nonzero chunk locations '''
        try:
            padded_index = self.indexloc.copy()
        except(AttributeError):
            print('nonetype')
        chunkcontent = self.content
        self.boundarycontent = set()
        T, H, W = self.T, self.H, self.W
        for t, i, j, v in chunkcontent:
            point_pad = {(t + 1, i, j), (t - 1, i, j), (t, min(i + 1, H), j), (t, max(i - 1, 0), j),
                         (t, i, min(j + 1, W)), (t, i, max(j - 1, 0))}

            if point_pad.issubset(self.indexloc) == False:  # the current content is a boundary element
                self.boundarycontent.add((t, i, j, v))
            padded_index = padded_index.union(point_pad)

        if self.pad > 1:  # pad extra layers around the chunk observations
            # there is max height, and max width, but there is no max time.
            for p in range(2, self.pad + 1):
                for t, i, j, v in chunkcontent:
                    padded_boundary_set = {(t + p, i, j), (t - p, i, j), (t, min(i + p, H), j),
                                           (t, max(i - p, 0), j), (t, i, min(j + p, W)), (t, i, max(j - p, 0))}
                    padded_index = padded_index.union(padded_boundary_set)
        return T, H, W, padded_index

    def conflict(self, c_):
        # TODO: check if the contents are conflicting.
        return False

    def empty_counts(self):
        self.count = 0
        self.parse = 0
        self.birth = None  # chunk creation time
        # # empty transitional counts
        # for chunkidx in list(self.adjacency.keys()):
        #     for dt in list(self.adjacency[chunkidx].keys()):
        #         self.adjacency[chunkidx][dt] = 0
        # for chunkidx in list(self.preadjacency.keys()):
        #     for dt in list(self.preadjacency[chunkidx].keys()):
        #         self.preadjacency[chunkidx][dt] = 0
        # delete transition matrix, otherwise, it might proliferate with unwanted chunks
        self.adjacency = {}
        self.preadjacency = {}

        return

    def concatinate(self, cR, check=True):
        if check:
            if self.check_adjacency(cR):
                clcrcontent = self.content | cR.content
                clcr = Chunk(list(clcrcontent), H=self.H, W=self.W)
                clcr.T = self.T + cR.T
                return clcr
            else:
                return None

        else:
            if len(self.ordered_content) == 1 and len(cR.ordered_content) == 1:
                clcrcontent = self.ordered_content[0] | cR.ordered_content[0]
                clcr = Chunk(list(clcrcontent), H=self.H, W=self.W)
                clcr.T = self.T + cR.T
                return clcr
            else:
                clcrcontent = self.ordered_content + cR.ordered_content
                clcr = Chunk([], ordered_content=clcrcontent, H=self.H, W=self.W)
                clcr.T = self.T + cR.T
                return clcr


    def average_content(self):
        # average the stored content with the sequence
        # calculate average deviation
        averaged_content = set()
        assert (len(self.matching_seq) > 0)
        for m in list(self.matching_seq.keys()):  # iterate through content points
            thispt = list(m)
            n_pt = len(self.matching_seq[m])
            otherpt0 = 0
            otherpt1 = 0
            otherpt2 = 0
            otherpt3 = 0
            for pt in self.matching_seq[m]:
                otherpt0 += pt[0] - thispt[0]
                otherpt1 += pt[1] - thispt[1]
                otherpt2 += pt[2] - thispt[2]
                otherpt3 += pt[3] - thispt[3]
            count = max(1, self.count)
            thispt[0] = int(thispt[0] + 1 / count * otherpt0 / n_pt)
            thispt[1] = int(thispt[1] + 1 / count * otherpt1 / n_pt)
            thispt[2] = int(thispt[2] + 1 / count * otherpt2 / n_pt)
            thispt[3] = int(thispt[3] + 1 / count * otherpt3 / n_pt)  # TODO: content may not need to be averaged

            if np.any(thispt) < 0:
                print("")
            averaged_content.add(tuple(thispt))

        self.content = averaged_content
        self.T = int(np.atleast_2d(np.array(list(self.content)))[:,0].max() + 1)  # those should be specified when joining a chunking graph
        self.get_index()
        self.get_index_padded()  # update boundary content
        return

    def variable_check_match(self, seq):  # a sequence matches any of its instantiated variables
        '''returns true if the sequence matches any of the variable instantiaions
        TODO: test this function with variables '''
        if len(self.includedvariables) == 0:
            return self.check_match(seq)
        else:
            match = []
            for ck in self.includedvariables:
                match.append(ck.variable_check_match(seq))
            return any(match)
        

    def check_match(self, seq):
        ''' Check explicit content match'''
        self.matching_seq = {}  # free up memory
        # key: chunk content, value: matching points
        # TODO: one can do better with ICP or Point Set Registration Algorithm.
        D = self.D

        def dist(m, pt):
            return (pt[0] - m[0]) ** 2 + (pt[1] - m[1]) ** 2 + (pt[2] - m[2]) ** 2 + (pt[3] - m[3]) ** 2

        def point_approx_seq(m, seq):  # sequence is ordered in time
            for pt in seq:
                if dist(m, pt) <= D:
                    if m in self.matching_seq.keys():
                        self.matching_seq[m].append(pt)
                    else:
                        self.matching_seq[m] = [pt]
                    return True
            return False

        n_match = 0
        for obs in list(self.content):  # find the number of observations that are close to the point
            if point_approx_seq(obs, seq):  # there exists something that is close to this observation in this sequence:
                n_match = n_match + 1

        if n_match / len(self.content) > self.matching_threshold:
            return True  # 80% correct
        else:
            return False
        



    def check_adjacency(self, cR):
        # dt: start_post - start_prev
        """Check if two chunks overlap/adjacent in their content and location"""
        cLidx = self.indexloc
        _,_,_, cRidx = cR.get_index_padded()
        intersect_location = cLidx.intersection(cRidx)
        if (
            len(intersect_location) > 0
        ):  # as far as the padded chunk and another is intersecting,
            return True
        else:
            return False

    def check_adjacency_approximate(self, cR, dt=0):
        # problematic implementation based on min and max of the boundaries.
        def overlaps(a, b):
            """
            Return the amount of overlap,
            between a and b. Bounds are exclusive.
            If >0, the number of bp of overlap
            If 0,  they are book-ended.
            If <0, the distance in bp between them
            """
            return min(a[1], b[1]) - max(a[0], b[0]) - 1

        """Check if two chunks overlap/adjacent in their content and location"""
        cLidx = self.indexloc
        cRidx = cR.indexloc
        intersect_location = cLidx.intersection(cRidx)

        Mcl, Mcr = np.array(list(cLidx)), np.array(list(cRidx))

        tl1, xl1, yl1 = Mcl.min(axis=0)
        tl2, xl2, yl2 = Mcl.max(axis=0)

        tr1, xr1, yr1 = Mcr.min(axis=0)
        tr2, xr2, yr2 = Mcr.max(axis=0)

        lap_t = overlaps((tl1 - self.pad, tl2 + self.pad), (dt + tr1 - self.pad, dt + tr2 + self.pad))
        lap_x = overlaps((xl1 - self.pad, xl2 + self.pad), (xr1 - self.pad, xr2 + self.pad))
        lap_y = overlaps((yl1 - self.pad, yl2 + self.pad), (yr1 - self.pad, yr2 + self.pad))

        if (lap_t > 0 and lap_x > 0 and lap_y > 0):
            return True
        else:
            return False

    def checksimilarity(self, chunk2):
        '''returns the minimal moving distance from point cloud chunk1 to point cloud chunk2'''
        pointcloud1, pointcloud2 = self.content.copy(), chunk2.content.copy()
        lc1, lc2 = len(pointcloud1), len(pointcloud2)
        # smallercloud = [pointcloud1,pointcloud2][np.argmin([lc1,lc2])]
        # match by minimal distance
        match = []
        minD = 0
        for x1 in pointcloud1:
            mindist = 1000
            minmatch = None
            # search for the matching point with the minimal distance
            if len(match) == min(lc1, lc2):
                break
            for x2 in pointcloud2:
                D = self.pointdistance(x1, x2)
                if D < mindist:
                    minmatch = (x1, x2)
                    mindist = D
            match.append(minmatch)
            minD = minD + mindist
            pointcloud2.pop(minmatch[1])
        return minD

    def pointdistance(self, x1, x2):
        ''' calculate the the distance between two points '''
        D = (x1[0] - x2[0]) * (x1[0] - x2[0]) + self.h * (x1[1] - x2[1]) * (x1[1] - x2[1]) + self.w * (
                    x1[2] - x2[2]) * (x1[2] - x2[2]) + self.v * (x1[0] - x2[0]) * (x1[0] - x2[0])
        return D

    def update_transition(self, chunk, dt, variable_adjacency_update = False):
        '''Update adjacency matrix connecting self to adjacent chunks with time distance dt
        Also update the adjacenc matrix of variables
        self: previous chunk
        chunk: post chunk
        '''
        #  self ---> chunk
        self.adjacency.setdefault(chunk.key, {}).setdefault(dt, 1)
        self.adjacency[chunk.key][dt] += 1
        #  self <--- chunk
        chunk.preadjacency.setdefault(self.key, {}).setdefault(dt, 1)
        chunk.preadjacency[self.key][dt] += 1

        if variable_adjacency_update:
            #######   update the transition between self, chunk, and their associated variables ########

            #   v   ---> v_c
            #  self ---> chunk
            for v in self.abstraction.values(): # update the transition for the parent variables of the chunk
                chunk.preadjacency.setdefault(v.key, {}).setdefault(dt, 1)
                chunk.preadjacency[v.key][dt] += 1

                v.adjacency.setdefault(chunk.key, {}).setdefault(dt, 1)
                v.adjacency[chunk.key][dt] += 1

                # update the transition amongst the variables
                for v_c in chunk.abstraction.values():
                    v.adjacency.setdefault(v_c.key,{}).setdefault(dt, 1)
                    v.adjacency[v_c.key][dt] += 1

            #   v   <--- v_c
            #  self <--- chunk
            for v_c in chunk.abstraction.values():
                v_c.preadjacency.setdefault(self.key, {}).setdefault(dt, 1)
                v_c.preadjacency[self.key][dt] += 1

                self.adjacency.setdefault(v_c.key, {}).setdefault(dt, 1)
                self.adjacency[v_c.key][dt] += 1

                # update the transition amongst the variables
                for v in self.abstraction.values():
                    v_c.preadjacency.setdefault(v.key, {}).setdefault(dt, 1)
                    v_c.preadjacency[v.key][dt] += 1
        return

    def contentagreement(self, content):
        if len(self.content) != len(content):
            return False
        else:  # sizes are the same
            return len(self.content.intersection(content)) == len(content)

    def get_pl(self, n_ans = 0):
        '''Obtain the parsing length needed to get from the parent chunk to this particular chunk '''

        if len(list(self.acl.keys())) == 0:
            return n_ans
        else:
            ancestor = list(self.acl.keys())[0]# TODO: what if there are multiple ancestors?
            return len(ancestor.cr) # the number of children of its right descendent (as so far parsing is implemented from the left to right)

    def get_rep_entropy(self, gt = False):
        '''The uncertainty carried via identifying this chunk to parse the sequene'''
        if len(self.includedvariables) == 0:
            return 0
        else:
            entropy = 0 # variables are assumed to be independent, therefore their entropies are additive
            for key, var in self.includedvariables.items():
                entropy = entropy + var.get_rep_entropy(gt=gt)
            self.entropy = entropy
            return entropy







# TODO: upon parsing, isinstance(51,Chunk) can be used to check whether something is a chunk or a variable

import random
import string


class Variable():
    """A variable can take on several contents"""

    # A code name unique to each variable
    def __init__(self, entailingchunks, count=1):  # how to define a variable?? a list of alternative
        ##################### Property Parameter ######################
        self.count = self.get_count(entailingchunks)
        self.key = self.get_variable_key()
        self.identificationfreq = 1 # the number of times that the variable entity is identified
        self.current_content = None # dynamic value, any of the entailing chunks that this variable is taking its value in

        ##################### Relational Parameter ######################
        self.adjacency = self.get_adjacency(entailingchunks)#should the adjaceny specific to individual variable instances, or as the entire variable? entire variable.
        self.preadjacency = self.get_preadjacency(entailingchunks)#should the adjaceny specific to individual variable instances, or as the entire variable? entire variable.

        self.entailingchunks = entailingchunks
        self.chunks = {} # chunks that this variable is a part of
        self.abstraction = {} # variables that this variable is a part of
        self.all_abstraction = set()  # all of the variable's abstractions
        self.chunk_probabilities = {}
        self.ordered_content = [self.key] #specify the order of chunks and variables
        self.vertex_location = self.get_vertex_location(entailingchunks)
        self.volume = self.get_average_explanable_volume(entailingchunks)


        # There is only variable relationship, but no relationship between variable and left/right parent/child,
        self.boundarycontent = set()
        self.D = 1
        self.T = 1

        self.cl = {}  # left decendent
        self.cr = {}  # right decendent
        self.acl = {} # left ancestor
        self.acr = {} # right ancestor


    def merge_two_variables(self, v):
        # when there are two variables that each entails the same chunks, merge the two variables into one

        self.count = self.count + v.count
        self.adjacency = self.merge_adjacency(self.adjacency, v.adjacency)
        self.preadjacency = self.preadjacency | v.preadjacency
        self.entailingchunks = self.entailingchunks | v.entailingchunks
        self.chunk_probabilities = self.merge_chunk_probabilities(self.chunk_probabilities, v.chunk_probabilities)
        self.all_abstraction = self.all_abstraction.union(v.all_abstraction)
        return

    def update_transition(self, chunk, dt):
        '''Update adjacency matrix connecting self to adjacent chunks with time distance dt
        Also update the adjacenc matrix of variables '''
        self.adjacency.setdefault(chunk.key, {}).setdefault(dt, 1)
        self.adjacency[chunk.key][dt] += 1

        chunk.preadjacency.setdefault(self.key, {}).setdefault(dt, 1)
        chunk.preadjacency[self.key][dt] += 1
        return


    def merge_adjacency(self, adj1, adj2):
        # merge two adjacency matrices from two variables
        for key,_ in adj2.items():
            adj1.setdefault(key, {})
            for dt, freq in adj2[key].items():
                adj1[key].setdefault(dt, 0)
                adj1[key][dt] += adj2[key][dt]
        return adj1

    def merge_chunk_probabilities(self, cp1, cp2):
        # merge two chunk probabilities from two variables
        for key, value in cp2.items():
            cp1.setdefault(key, 0)
            cp1[key] += cp2[key]
        return cp1


    def sample_current_content(self):
        '''sample one of the entailing chunks as the current content of the variable'''
        self.current_content = np.random.choice(list(self.chunk_probabilities.keys()), 1, p = list(self.chunk_probabilities.values()))
        return

    def sample_content(self):
        '''sample one of the entailing chunks as the current content of the variable'''
        counts = list(self.chunk_probabilities.values())
        ps = [c/sum(counts) for c in counts]
        idx = np.random.choice(np.arange(len(list(self.chunk_probabilities.keys()))), 1,
                               p=ps)[0]

        sampled_chunk_name = list(self.chunk_probabilities.keys())[idx]
        return self.entailingchunks[sampled_chunk_name] # returns a list containing sets and strings as its items

    def substantiate_chunk_probabilities(self):
        """Based on the entailing chunks, substantiate the chunk probabilities for sampling and other proposes """
        for ck in self.entailingchunks.values():
            self.chunk_probabilities[ck.key] = ck.count
        return


    def get_average_explanable_volume(self, entailingchunks):
        """Evalaute the average explanatory volume based on one parsing of such variable"""
        if type(entailingchunks) == set:
            temp = list(entailingchunks)
        else:
            temp = list(entailingchunks.values())
        fs = []
        vs = []
        for ck in temp:
            fs.append(ck.count)
            vs.append(ck.volume)
        if sum(fs)>0:
            ps = [f/sum(fs) for f in fs]
        else:
            ps = [0]*len(fs)
        return sum([p*v for p,v in zip(ps,vs)])

    def get_count(self, entailingchunks):

        if type(entailingchunks) == set:
            count = 0
            for ck in list(entailingchunks):
                try:
                    count = count + ck.count
                except(TypeError):
                    print()
            return count
        else:
            count = 0
            for ck in entailingchunks.values():
                try:
                    count = count + ck.count
                except(TypeError):
                    print()
            return count

    def update(self):  # when any of its associated chunks are identified
        # if varinstance in self.content:
        #     self.count[varinstance] = self.count[varinstance] + 1
        self.count += 1
        return

    def get_vertex_location(self, entailingchunks):

        if type(entailingchunks) == set:
            temp = list(entailingchunks)
        else:
            temp = list(entailingchunks.values())

        xs = 0
        ys = 0
        for ck in temp:
            x,y = ck.vertex_location
            xs = xs + x
            ys = ys + y
        return xs/len(entailingchunks), ys/len(entailingchunks)

    def get_adjacency(self, entailingchunks):
        if type(entailingchunks) == set:
            temp = list(entailingchunks)
        else:
            temp = list(entailingchunks.values())
        # I think we might not need it
        adjacency = {}
        # entailingchunks = set(cg.chunks[item] for item in entailingchunks)
        for chunk in temp:
            for cr in chunk.adjacency:
                if cr in adjacency.keys():
                    for dt in chunk.adjacency[cr]:
                        if dt in list(adjacency[cr].keys()):
                            adjacency[cr][dt] = adjacency[cr][dt] + chunk.adjacency[cr][dt]
                        else:
                            adjacency[cr][dt] = chunk.adjacency[cr][dt]
                else:
                    adjacency[cr] = {}
                    for dt in chunk.adjacency[cr]:
                        adjacency[cr][dt] = chunk.adjacency[cr][dt]
        return adjacency

    def get_preadjacency(self, entailingchunks):
        if type(entailingchunks) == set:
            temp = list(entailingchunks)
        else:
            temp = list(entailingchunks.values())

        preadjacency = {}
        dts = set()
        for chunk in temp:
            for cl in chunk.preadjacency:
                if cl in preadjacency.keys():
                    for dt in chunk.preadjacency[cl]:
                        if dt in list(preadjacency[cl].keys()):
                            preadjacency[cl][dt] = preadjacency[cl][dt] + chunk.preadjacency[cl][dt]
                        else:
                            preadjacency[cl][dt] = chunk.preadjacency[cl][dt]
                else:
                    preadjacency[cl] = {}
                    for dt in chunk.preadjacency[cl]:
                        preadjacency[cl][dt] = chunk.preadjacency[cl][dt]
        return preadjacency



    def get_N_transition(self, dt):
        # todo: make nonozero
        N = 0
        for chunk in self.adjacency:
            if dt in self.adjacency[chunk]:
                N = N + self.adjacency[chunk][dt]
        return N

    def check_adjacency(self, cR):
        """ Check the adjacency between variable and cR as an observation """
        return any([_ck.check_adjacency(_ck, cR) for _ck in self.content])

    def check_match(self, seq):
        """check whether this variable is in sequence"""
        return any([_ck.check_match(_ck, seq) for _ck in self.content])

    def contentagreement(self, content):
        # if the content agree with any of the chunks within the varible
        pass

    def empty_counts(self):
        self.count = 0
        self.parse = 0
        self.birth = None  # chunk creation time
        # # empty transitional counts
        # for chunkidx in list(self.adjacency.keys()):
        #     for dt in list(self.adjacency[chunkidx].keys()):
        #         self.adjacency[chunkidx][dt] = 0
        # for chunkidx in list(self.preadjacency.keys()):
        #     for dt in list(self.preadjacency[chunkidx].keys()):
        #         self.preadjacency[chunkidx][dt] = 0
        # delete transition entry
        self.adjacency = {}
        self.preadjacency = {}
        self.identificationfreq = 0
        return


    def get_variable_key(self):
        length = 4
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))+'_'
        return result_str

    def check_variable_match(self, seqc):
        '''Check any of variables included in the chunk is consistent with observations of the sequence copy'''
        # for obj in self.content: # obj can be chunk, or variable
        pass

    def get_rc(self):
        ''' Evaluate the representation complexity
        rc: the encoding cost of distinguishing the entailing variables from its parent variable
         returns the minimal encoding length to distinguish the entailing variables/chunks from this variable
         '''
        freq = np.array([ck.count for ck in self.entailingchunks.values() if ck.count != 0])
        ps = freq / freq.sum()
        return -np.sum(np.log2(ps))


    def get_rep_entropy(self, gt = False):
        """Obtain the representation entropy of a variable"""
        if gt:
            ps = np.array(list(self.chunk_probabilities.values()))
            return -np.sum(ps * np.log2(ps))
        else:
            freq = np.array([ck.count for ck in self.entailingchunks.values() if ck.count != 0])
            ps = freq / freq.sum()
            return -np.sum(ps * np.log2(ps))


    def get_entailmentcount(self):
        return sum([ck.count for ck in self.entailingchunks.values()])
