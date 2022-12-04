# -------------------------- Binary Search Tree ------------------------------

# node class
class BSTNode:
    
    # initialise
    def __init__(self, info):
        self.info = info
        self.left = None
        self.right = None
        self.level = None
    
    # node printing
    def __str__(self):
        return str(self.info)

# tree class
class BinarySearchTree:
    
    # initialise
    def __init__(self): 
        self.root = None
    
    # insert value
    def insert(self, val):  
        # input: value to add
        if self.root == None:
            self.root = BSTNode(val)
        else:
            current = self.root
            while True:
                if val < current.info: # move to left sub-tree
                    if current.left:
                        current = current.left # root moved
                    else:
                        current.left = BSTNode(val) # left init
                        break
                elif val > current.info: # move to right sub-tree
                    if current.right:
                        current = current.right # root moved
                    else:
                        current.right = BSTNode(val) # right init
                        break
                else:
                    break # value exists
    
    # check BST
    def checkBST(self, m, M):
        # input: global minima and maxima (bounds)
        if self.root is None:
            return True
        if m < self.root.data and self.root.data < M:
            return self.checkBST(self.root.left,m,self.root.data) and self.checkBST(self.root.right,self.root.data,M)
        return False
    
    # traversal
    def traverse(self, order):
        # input: type of traversal from among 'PRE', 'IN' and 'POST'
        def preOrder(root):
            print(root.info, end = ' ')
            if root.left != None:
                preOrder(root.left)
            if root.right != None:
                preOrder(root.right)
        def inOrder(root):
            if root.left != None:
                inOrder(root.left)
            print(root.info, end = ' ')
            if root.right != None:
                inOrder(root.right)
        def postOrder(root):
            if root.left != None:
                postOrder(root.left)
            if root.right != None:
                postOrder(root.right)
            print(root.info, end = ' ')
        if order == 'PRE':
            preOrder(self.root)
        elif order == 'IN':
            inOrder(self.root)
        elif order == 'POST':
            postOrder(self.root)
    
    # height of tree
    def height(self, root):
        # input: root to find height for
        if root.left == None and root.right == None:
            return 0
        elif root.right == None:
            return 1 + self.height(root.left)
        elif root.left == None:
            return 1 + self.height(root.right)
        else:
            return 1 + max(self.height(root.left),self.height(root.right))
    
    # lowest common ancestor
    def lca(self, v1, v2):
        # input: values to check
        root = self.root
        while True:
            if root.left != None and v1 < root.info and v2 < root.info:
                root = root.left
            elif root.right != None and v1 > root.info and v2 > root.info:
                root = root.right
            else:
                return root