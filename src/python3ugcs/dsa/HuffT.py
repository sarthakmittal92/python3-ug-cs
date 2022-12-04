import queue as Queue

# ---------------------------------- Huffman Tree ------------------------------

cntr = 0
# node class
class HuffmanNode:
    
    # initialise
    def __init__(self, freq, data):
        self.freq = freq
        self.data = data
        self.left = None
        self.right = None
        global cntr
        self._count = cntr
        cntr += 1
    
    # comparator
    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq < other.freq
        return self._count < other._count

# tree class
class HuffmanTree:
    
    # initialise
    def __init__(self):
        self.root = None
    
    # tree builder
    def huffman(self, freq):
        # input: frequency dictionary
        q = Queue.PriorityQueue()        
        for key in freq:
            q.put((freq[key], key, HuffmanNode(freq[key], key)))
        while q.qsize() != 1:
            a = q.get()
            b = q.get()
            obj = HuffmanNode(a[0] + b[0], '\0' )
            obj.left = a[2]
            obj.right = b[2]
            q.put((obj.freq, obj.data, obj))            
        root = q.get()
        root = root[2]
        self.root = root
    
    # hidden DFS
    def dfs_hidden(self, obj, already, code_hidden):
        if(obj == None):
            return
        elif(obj.data != '\0'):
            code_hidden[obj.data] = already   
        self.dfs_hidden(obj.right, already + '1', code_hidden)
        self.dfs_hidden(obj.left, already + '0', code_hidden)
    
    # decoder
    def decodeHuff(self, s):
        # input: string to decode
        temp = self.root
        s = list(s)
        while len(s) > 0:
            if temp.left == None and temp.right == None:
                print(temp.data, end = '')
                temp = self.root
            if s.pop(0) == '0':
                temp = temp.left
            else:
                temp = temp.right
        print(temp.data)