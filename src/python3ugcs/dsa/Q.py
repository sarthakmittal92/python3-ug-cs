import heapq as hq

# ------------------------------ My Queue -------------------------------

# queue class
class MyQueue:
    
    # initialise
    def __init__(self):
        self.head = None
        self.tail = []
    
    # enqueue
    def enqueue(self, x):
        if self.head == None:
            self.head = x
        else:
            self.tail.append(x)
    
    # dequeue
    def dequeue(self):
        if self.head != None:
            if self.tail != []:
                self.head = self.tail.pop(0)
            else:
                self.head = None
    
    # front element
    def printFront(self):
        print(self.head)

# --------------------------- Binary Heap Priority Queue --------------------------------

# qheap class
class QHeap:
    
    # initialise
    def __init__(self):
        self.A = []
    
    # length
    def length(self):
        return len(self.A)
    
    # insert value
    def insert(self, val):
        # input: value to add
        hq.heappush(self.A,val)
    
    # fetch minimum
    def getMin(self):
        return self.A[0]
    
    # fetch last
    def pop(self):
        return hq.heappop(self.A)
    
    # delete value
    def delete(self, val):
        # input: value to delete
        idx = self.A.index(val)
        self.A[idx] = self.A[-1]
        self.A.pop()
        if idx < len(self.A):
            hq._siftup(self.A,idx)
            hq._siftdown(self.A,0,idx)