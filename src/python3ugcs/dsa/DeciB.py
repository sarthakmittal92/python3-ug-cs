from bisect import bisect_left
import math

# ---------------------------------- DeciBinary -------------------------------

# number class
class DeciBinary:
    
    # initialise
    def __init__(self):
        self.COUNT = [[1, 1]]
        self.sums = [1]
        self.generateMinDigits()
    
    # generate minimum digit numbers
    def generateMinDigits(self, expMax = 20):
        self.minDigits = [0] 
        for p in range(expMax):
            self.minDigits.append(9 * 2 ** p + self.minDigits[-1])
    
    # find appropriate size number
    def calculateMinDigits(self, n):
        return bisect_left(self.minDigits, n)
    
    # find decibinary at position x
    def decibinaryNumbers(self, x):
        if x > self.sums[-1]:
            self.extend(x)
        return self.get(x)
    
    # find and add number at position
    def extend(self, num):
        import math
        n = len(self.COUNT)
        while self.sums[-1] < num:
            minDigits = self.calculateMinDigits(n)
            maxDigits = math.floor(math.log(n, 2)) + 1
            self.COUNT.append([0] * (maxDigits + 1))
            for m in range(minDigits, maxDigits + 1):
                self.COUNT[n][m] = self.COUNT[n][m - 1]
                for d in range(1, 10):
                    remainder = n - d * 2 ** (m - 1)
                    if remainder >= 0:
                        self.COUNT[n][m] += self.COUNT[remainder][min(m - 1, len(self.COUNT[remainder]) - 1)]
                    else: 
                        break
            self.sums.append(self.sums[-1] + self.COUNT[-1][-1])
            n += 1
    
    # get decibinary at position x        
    def get(self, x):
        if x == 1:
            return 0
        n = bisect_left(self.sums, x)
        rem1 = x - self.sums[n - 1]
        m = bisect_left(self.COUNT[n], rem1)
        rem2 = rem1 - self.COUNT[n][m - 1]
        return self.reconstruct(n, m, rem2)
    
    # adjust decibinary list
    def reconstruct(self, n, m, rem, partial = 0):
        if m == 1:
            return partial + n
        skipped = 0
        for k in range(not partial, 10):
            value = k * 2 ** (m - 1)
            smaller = n - value
            temp = min(len(self.COUNT[smaller]) - 1, m - 1)
            skipped += self.COUNT[smaller][temp]
            if skipped >= rem:
                partial += k * 10 ** (m - 1)
                rem1 = rem - (skipped - self.COUNT[smaller][temp])
                return self.reconstruct(smaller, temp, rem1, partial)