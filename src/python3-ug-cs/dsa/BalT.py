from itertools import groupby

# -------------------------------- Balance Tree ------------------------------

# node class
class ValNode:
    
    # initialise
    def __init__(self, i, val):
        self.idx = i
        self.val = val
        self.children = []

# tree class
class BalanceTree:
    
    # initialise
    def __init__(self):
        self.nodes = []
    
    # build
    def build(self, values, edges):
        self.nodes = [ValNode(i,v) for i, v in enumerate(values)]
        for s, d in edges:
            self.nodes[s - 1].children.append(self.nodes[d - 1])
            self.nodes[d - 1].children.append(self.nodes[s - 1])
    
    # root-ify
    def rootify(self):
        # traverse upto depth
        def uptoDepth(node, depth):
            while node.d > depth:
                if node.up.d <= depth:
                    node = node.up
                else:
                    node = node.p
            return node
        root = self.nodes[0]
        root.p, root.up, root.d = None, None, 0
        flattened = []
        flattened.append(root)
        i = 0
        while i < len(flattened):
            node = flattened[i]
            depth = node.d + 1
            moveUp = uptoDepth(node,depth & (depth - 1))
            for c in node.children:
                c.p, c.up, c.d = node, moveUp, depth
                c.children.remove(node)
                flattened.append(c)
            i += 1
        for node in reversed(flattened):
            node.tv = node.val + sum([c.tv for c in node.children])
    
    # balance
    def balance(self):
        def uptoTotal(node, val):
            try:
                while node.tv < val:
                    if node.p is None:
                        return None
                    if node.up.tv <= val:
                        node = node.up
                    else:
                        node = node.p
                if node.tv == val:
                    return node
                return None
            except Exception:
                return None
        if len(self.nodes) == 1:
            return -1
        self.rootify()
        total = self.nodes[0].tv
        temp = ValNode(None,None)
        temp.tv = 0
        sortedNodes = []
        for _, g in groupby(sorted([temp] + self.nodes, key = lambda x: x.tv), lambda x: x.tv):
            sortedNodes.append(list(g))
        total = self.nodes[0].tv
        for i0, n in enumerate(sortedNodes):
            if 3 * n[0].tv >= total:
                break
        else:
            assert False
        i1 = i0 - 1
        for i0 in range(i0,len(sortedNodes)):
            h = sortedNodes[i0][0].tv
            l = sortedNodes[i1][0].tv
            while 2 * h + l > total:
                if l == 0:
                    return -1
                if (total - l) % 2 == 0:
                    x = (total - l) // 2
                    for ln in sortedNodes[i1]:
                        if uptoTotal(ln,x + l):
                            return x - l
                i1 -= 1
                l = sortedNodes[i1][0].tv
            if len(sortedNodes[i0]) > 1:
                return 3 * h - total
            hn = sortedNodes[i0][0]
            if 2 * h + l == total:
                for ln in sortedNodes[i1]:
                    if uptoTotal(ln,h) != hn:
                        return h - l
            y = total - (2 * h)
            if uptoTotal(hn,2 * h) or uptoTotal(hn,h + y):
                return h - y