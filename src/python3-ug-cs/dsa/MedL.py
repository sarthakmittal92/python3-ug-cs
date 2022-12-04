# -------------------------------- Median List --------------------------------

# list class
class MedianList:
    
    # initialise
    def __init__(self):
        self.A = []
    
    # insert value
    def insert(self, v):
        # input: value to add
        s, e = 0, len(self.A)
        while s < e:
            mid = (s + e) // 2
            if self.A[mid] <= v:
                s = mid + 1
            else:
                e = mid
        self.A.insert(s,v)
    
    # fetch median
    def getMedian(self):
        k = len(self.A) // 2
        if len(self.A) % 2 == 1:
            return self.A[k]
        return (self.A[k] + self.A[k - 1]) / 2