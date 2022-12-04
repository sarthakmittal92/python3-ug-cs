from collections import deque
from .Q import QHeap
import math

# ------------------------------------ Graph ----------------------------------

# graph class
class Graph:
    
    # initialise
    def __init__(self):
        self.n = 0
        self.nodes = set()
        self.m = 0
        self.src  = []
        self.dest = []
        self.w = []
    
    # add edges (and their nodes)
    def addEdge(self, s, d, w):
        self.nodes.add(s)
        self.nodes.add(d)
        self.n = len(self.nodes)
        self.m += 1
        self.src.append(s)
        self.dest.append(d)
        self.w.append(w)
    
    # build
    def build(self, src, dest, w):
        for i in range(len(src)):
            self.addEdge(src[i],dest[i],w[i])
    
    # color-build
    def colorBuild(self, src, dest, ids):
        for i in range(len(src)):
            self.addEdge(src[i],dest[i],1)
        self.colorMap = {}
        for i, c in enumerate(ids):
            self.colorMap.setdefault(c,[])
            self.colorMap[c].append(i + 1)
    
    # tuple-weight form of edges
    def edgeMap(self):
        # format: {(s,d):w}
        self.edges = {(self.src[i],self.dest[i]):self.w[i] for i in range(self.m)}
        # self.edges.update({(self.dest[i],self.src[i]):self.w[i] for i in range(self.m)})
    
    # triplet set form of edges
    def edgeSet(self):
        # format: [(s,d,w)]
        self.edges = [(self.src[i],self.dest[i],self.w[i]) for i in range(self.m)]
        # self.edges.extend([(self.dest[i],self.src[i],self.w[i]) for i in range(self.m)])
    
    # adjacency list form of edges
    def edgeList(self):
        # format: outEdges[s] = (d,w) and inEdges[d] = (s,w)
        self.outEdges = [[] * (self.n + 1)]
        self.inEdges = [[] * (self.n + 1)]
        for i in range(self.m):
            self.outEdges[self.src[i]].append((self.dest[i],self.w[i]))
            self.inEdges[self.dest[i]].append((self.src[i],self.w[i]))
    
    # adjacency matrix form of edges
    def edgeMatrix(self):
        # format: E[s][d] = w and E[d][s] = w
        self.E = [[] * (self.n + 1) for _ in range(self.n + 1)]
        for i in range(self.m):
            self.E[self.src[i]][self.dest[i]] = self.w[i]
            # self.E[self.dest[i]][self.src[i]] = self.w[i]
    
    # matrix to adjacency list
    def edgeMatBuild(self, grid):
        # max region
        def getMaxRegion(i, j):
            CC = set()
            if i > 0:
                if self.grid[i - 1][j] == 1:
                    CC.add((i - 1,j))
                if j > 0:
                    if self.grid[i - 1][j - 1] == 1:
                        CC.add((i - 1,j - 1))
                if j < self.M - 1:
                    if self.grid[i - 1][j + 1] == 1:
                        CC.add((i - 1,j + 1))
            if i < self.N - 1:
                if self.grid[i + 1][j] == 1:
                    CC.add((i + 1,j))
                if j > 0:
                    if self.grid[i + 1][j - 1] == 1:
                        CC.add((i + 1,j - 1))
                if j < self.M - 1:
                    if self.grid[i + 1][j + 1] == 1:
                        CC.add((i + 1,j + 1))
            if j > 0:
                if self.grid[i][j - 1] == 1:
                    CC.add((i,j - 1))
            if j < self.M - 1:
                if self.grid[i][j + 1] == 1:
                    CC.add((i,j + 1))
            return CC
        self.grid = grid
        self.N = len(grid)
        self.M = len(grid[0])
        self.edges = {}
        for i in range(self.N):
            for j in range(self.M):
                if self.grid[i][j] == 1:
                    self.edges[(i,j)] = getMaxRegion(i,j)
    
    # print graph
    def printer(self):
        self.edgeMap()
        for (s, d), w in self.edges.items():
            print('(' + str(s) + ', ' + str(d) + ') -> ' + str(w))
    
    # BFS and DFS
    def search(self, s, mode):
        # input: start vertex and mode of search from among 'BFS' and 'DFS'
        # paradigm: Dynamic Programming
        self.edgeList()
        visited = {i:False for i in self.nodes}
        dp = []
        Q = deque()
        if mode == 'BFS':
            Q.append(s)
        else:
            Q.appendleft(s)
        visited[s] = True
        while len(Q) > 0:
            u = Q.popleft()
            dp.append(u)
            for v, _ in self.outEdges[u]:
                if not visited[v]:
                    if mode == 'BFS':
                        Q.append(v)
                    else:
                        Q.appendleft(v)
                    visited[v] = True
        return dp
    
    # connected components (DFS)
    def CC(self, s):
        sourceSet = [s]
        visited = set()
        while len(sourceSet) > 0:
            u = sourceSet.pop()
            if u not in visited:
                visited.add(u)
                sourceSet.extend(self.edges[u] - visited)
        return len(visited)
    
    # largest connected component
    def expand(self):
        CCsizes = []
        for i in range(self.N):
            for j in range(self.M):
                if self.grid[i][j] == 1:
                    CCsizes.append(self.CC((i,j)))
        return max(CCsizes)
    
    # disconnect machines
    def disconnect(self, n, machineCities):
        def swapGroup(groups, cityMap, g1, g2):
            for c in groups[g1]:
                cityMap[c] = g2
                groups[g2].append(c)
            del groups[g1]
        self.edgeMap()
        self.edges = dict(sorted(self.edges.items(), key = lambda x: x[1], reverse = True))
        cityMap = {}
        machines = {}
        for city in machineCities:
            machines[city] = True
            cityMap[city] = city
        res = 0
        groups = {}
        for i in range(n):
            groups[i] = [i]
            cityMap[i] = i
        for edge, weight in self.edges.items():
            c1, c2 = edge[0], edge[1]
            g1, g2 = cityMap[c1], cityMap[c2]
            if g1 == g2:
                continue
            if g1 in machines:
                if g2 in machines:
                    res += weight
                else:
                    swapGroup(groups,cityMap,g2,g1)
            else:
                swapGroup(groups,cityMap,g1,g2)
        return res
    
    # Dijkstra
    def dijkstra(self, s):
        # input: source node
        # paradigm: Greedy
        H = QHeap()
        self.edgeList()
        gd = [-1 for _ in range(self.n)]
        H.insert((0,s))
        visited = {i:False for i in self.nodes}
        while H.length() > 0:
            c, u = H.pop()
            if not visited[u]:
                visited[u] = True
                gd[u] = c
                for v, w in self.outEdges[u]:
                    if not visited[v]:
                        H.insert((c + w,v))
        return gd
    
    # Kruskal's MST
    def kruskal(self):
        # input: None
        # paradigm: Greedy
        def find(p, i):
            if p[i] == i:
                return i
            return find(p,p[i])
        def union(p, r, x, y):
            xroot = find(p,x)
            yroot = find(p,y)
            if r[xroot] < r[yroot]:
                p[xroot] = yroot
            else:
                p[yroot] = xroot
                if r[xroot] == r[yroot]:
                    r[xroot] += 1
        self.edgeSet()
        gd = []
        i, e = 0, 0
        p, r = [], []
        for node in self.nodes:
            p.append(node)
            r.append(0)
        while e < self.n - 1:
            u, v, w = self.edges[i]
            i += 1
            x = find(p,u)
            y = find(p,v)
            if x != y:
                e += 1
                gd.append(self.edges[i])
                union(p,r,x,y)
        MST = set()
        minCost = 0
        for u, v, w in gd:
            MST.add((u,v))
            minCost += w
        return minCost
    
    # Prim's MST
    def prims(self, start):
        # input: start vertex
        # paradigm: Greedy
        def helper(edge, T):
            return (edge[0] in T) ^ (edge[1] in T)
        self.edgeMap()
        MST = set()
        MST.add(start)
        weights = []
        while len(MST) < self.n:
            newEdge = [(-1,-1),100000]
            for edge, weight in self.edges:
                if helper(edge, MST) and weight < newEdge[1]:
                    newEdge[0] = edge
                    newEdge[1] = weight
            MST.add(newEdge[0][0])
            MST.add(newEdge[0][1])
            weights.append(newEdge[1])
        return sum(weights)

    # Floyd-Warshall (all-pair-shortest-paths)
    def floydWarshall(self, Q):
        # input: queries (x and y)
        # paradigm: Greedy
        def helper(n, edges, s):
            sourceSet = set()
            sourceSet.add(s)
            gd = [10000] * (n + 1)
            gd[s] = 0
            while len(sourceSet) > 0:
                outgoing = {edge:weight for edge, weight in edges.items() if edge[0] in sourceSet}
                neighbourhood = set()
                for path, w in outgoing.items():
                    s, d = path
                    temp = gd[s] + w
                    if temp < gd[d]:
                        gd[d] = temp
                        neighbourhood.add(d)
                sourceSet = neighbourhood
            return [-1 if i >= 10000 else i for i in gd]
        self.edgeMap()
        sourceSet = set([x for x, _ in Q])
        gd = [[] for _ in range(self.n + 1)]
        for s in sourceSet:
            gd[s] = helper(self.n,self.edges,s)
        for x, y in Q:
            print(gd[x][y])
    
    # colored shortest paths
    def findColouredShortest(self, val):
        # input: color to match
        # paradigm: Greedy
        def helper(n, edges, s):
            import math
            sourceSet = set()
            sourceSet.add(s)
            visited = set()
            visited.add(s)
            gd = [10000] * (n + 1)
            gd[s] = 0
            while len(sourceSet) > 0:
                visited = visited | sourceSet
                outgoing = {edge:weight for edge, weight in edges.items() if edge[0] in sourceSet and edge[1] not in visited}
                neighbourhood = set()
                for path, w in outgoing.items():
                    s, d = path
                    temp = gd[s] + w
                    if temp < gd[d]:
                        gd[d] = temp
                        neighbourhood.add(d)
                sourceSet = neighbourhood
            return [math.inf if i >= 10000 else i for i in gd]
        if val not in self.colorMap:
            return -1
        l = self.colorMap[val]
        Q = set()
        for u in l:
            for v in l:
                if u != v:
                    Q.add((u,v))
        if len(Q) < 2:
            return -1
        self.edgeMap()
        sourceSet = set([x for x, _ in Q])
        gd = [[] for _ in range(self.n + 1)]
        for s in sourceSet:
            gd[s] = helper(self.n,self.edges,s)
        return min([gd[x][y] for x, y in Q if gd[x][y] > 0])
    
    # Bus fare problem
    def getCost(self):
        # input: None
        # paradigm: Greedy
        x = 1
        y = self.n
        I = math.inf
        edges = [[] for _ in range(self.n + 1)]
        for i in range(len(self.src)):
            edges[self.src[i]].append((self.dest[i],self.w[i]))
            edges[self.dest[i]].append((self.src[i],self.w[i]))
        gd = [I] * (self.n + 1)
        gd[x] = 0
        source = set()
        source.add(x)
        while len(source) > 0:
            neighbourhood = set()
            for s in source:
                outgoing = edges[s]
                for d, w in outgoing:
                    temp = max(gd[s],w)
                    if temp < gd[d]:
                        gd[d] = temp
                        neighbourhood.add(d)
            source = neighbourhood
        if gd[self.n] >= I:
            print('NO PATH EXISTS')
        else:
            print(gd[self.n])

# ---------------------------------- Node Graph ------------------------------

# node class
class GraphNode:
    
    # initialise
    def __init__(self, i):
        # input: index of node
        self.idx = i
        self.cost = -1
        self.nbhd = set()

# graph class
class NodeGraph:
    
    # initialise
    def __init__(self, n):
        # input: number of nodes
        self.nodes = [GraphNode(i) for i in range(n)]
    
    # add edges
    def addEdge(self, s, d):
        # input: source and destination nodes
        self.nodes[s].nbhd.add(d)
        self.nodes[d].nbhd.add(s)
    
    # find distances (BFS)
    def findCost(self, s):
        # input: source node
        # paradigm: Search
        sourceSet = set()
        sourceSet.add(s)
        w = 6
        currCost = 0
        while len(sourceSet) > 0:
            neighbourhood = set()
            for s in sourceSet:
                node = self.nodes[s]
                if node.cost == -1:
                    neighbourhood.update(node.nbhd)
                    self.nodes[s].cost = currCost
            sourceSet = neighbourhood
            currCost += w
        return ' '.join([str(node.cost) for node in self.nodes if node.cost != 0])