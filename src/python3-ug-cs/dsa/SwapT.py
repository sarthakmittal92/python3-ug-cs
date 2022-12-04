from collections import deque

# ------------------------------------ Swap Tree ------------------------------

# node class
class SwapNode:
    
    # initialise
    def __init__(self, i):
        # input: index of node
        self.l = None
        self.r = None
        self.i = i

# tree class
class SwapTree:
    
    # initialise
    def __init__(self, n):
        # input: number of nodes
        self.n = n
        self.nodes = [None] * (n + 1)
    
    # build using 2-D array by modelling it as a heap
    def build(self, indexes):
        for i in range(1,self.n + 1):
            node = SwapNode(i)
            node.left, node.right = [max(v, 0) for v in indexes[i - 1]]
            self.nodes[i] = node
        for node in self.nodes[1:]:
            l, r = self.nodes[node.left], self.nodes[node.right]
            node.l, node.r = l, r
        self.root = self.nodes[1]
    
    # in-order traversal using deque
    def inOrder(self):
        stack = deque([self.root])
        visited = set()
        res = []
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if node.i in visited:
                res.append(node.i)
                continue
            visited.add(node.i)
            stack.append(node.r)
            stack.append(node)
            stack.append(node.l)
        return res
    
    # swap nodes at levels k, 2k, 3k, ..
    def swap(self, k):
        q = deque([(self.root, 1)])
        while q:
            node, level = q.popleft()
            if node is None:
                continue
            if level % k == 0:
                node.l, node.r = node.r, node.l
            q.append((node.l, level + 1))
            q.append((node.r, level + 1))