# --------------------------------- Suffix Trie --------------------------------

# trie class
class Trie:
    
    # initialise
    def __init__(self):
        self.T = {}
    
    # finder
    def find(self, root, c):
        # input: root of trie and char to find
        return (c in root)
    
    # insert string
    def insert(self, s):
        # input: string to insert
        root = self.T
        for c in s:
            if not self.find(root,c):
                root[c] = {}
            root = root[c]
            root.setdefault('#',0)
            root['#'] += 1
    
    # check if string is a prefix
    def checkPrefix(self, s):
        # input: string to check
        root = self.T
        for idx, char in enumerate(s):
            if char not in root:
                if idx == len(s) - 1:    
                    root[char] = '#'
                else:
                    root[char] = {}
            elif root[char] == '#' or idx == len(s) - 1:
                return True
            root = root[char]
        return False
    
    # count prefix matches
    def countPrefix(self, s):
        # input: prefix to match
        found = True
        root = self.T
        for c in s:
            if self.find(root,c):
                root = root[c]
            else:
                found = False
                break
        if found:
            return root['#']
        return 0