from collections import Counter
import math
from bisect import bisect_left, bisect_right, bisect, insort

##################################### Algorithms ################################

# --------------------------------- Brute Force -------------------------------

# Check magic squares
def formingMagicSquare(s):
    # input: 2-D array
    # paradigm: Brute Force
    acceptables = ['834159672','618753294','294753618','672159834']
    acceptables += [''.join(list(s)[-1::-1]) for s in acceptables]
    def helper (grid):
        s = ''
        for row in grid:
            for col in row:
                s += str(col)
        return s
    seq = helper(s)
    return seq in acceptables

# ----------------------------- Pointer + Accumulator ---------------------------

# Bracket balance
def isBalanced(s):
    # input: bracket string
    # paradigm: Pointer + Accumulator
    openers = ['(','{','[']
    closers = [')','}',']']
    stack = []
    for c in s:
        if c in openers:
            stack.append(c)
        else:
            if len(stack) == 0:
                return 'NO'
            if openers.index(stack[-1]) == closers.index(c):
                stack.pop()
            else:
                return 'NO'
    if len(stack) != 0:
        return 'NO'
    return 'YES'

# Maximum sub-array/sub-sequence sums
def maxSubSums(arr):
    # input: array
    # paradigm: Pointer + Accumulator
    maxSubArraySum = 0
    maxSubSequenceSum = 0
    currSum = 0
    M = max(arr)
    if M < 0:
        return [M,M]
    for i in range(len(arr)):
        if arr[i] > 0:
            maxSubSequenceSum += arr[i]
            currSum += arr[i]
        else:
            if currSum + arr[i] < 0:
                currSum = 0
            else:
                currSum += arr[i]
        maxSubArraySum = max(maxSubArraySum,currSum)
    return [maxSubArraySum,maxSubSequenceSum]

# Maximum sub-array sum (with modulus)
def maximumSum(a, m):
    # input: array and modular number
    # output: maximum modular sub-array sum
    # paradigm: Pointer + Accumulator
    cumuSum = []
    maxSubarraySum = 0
    currSum = 0
    for i in range(len(a)):
        currSum = (currSum + a[i]) % m
        pos = bisect(cumuSum, currSum)
        if pos == i:
            d = 0
        else:
            d = cumuSum[pos]
        maxSubarraySum = max(maxSubarraySum, (currSum + m - d) % m)
        insort(cumuSum, currSum)
    return maxSubarraySum

# -------------------------------- Two Pointers ---------------------------------

# Max permissible area (water container)
def maxArea(height):
    # input: height array
    # paradigm: Two-Pointers
    st = 0
    en = len(height) - 1
    A = (en - st) * min(height[st], height[en])
    while st < en:
        if height[st] < height[en]:
            st += 1
        else:
            en -= 1
        A = max(A, (en - st) * min(height[st], height[en]))
    return A

# Max permissible area (buildings with foundation limit)
def largestRectangle(h):
    # inout: array of heights
    # paradigm: Two-Pointers
    l = len(h)
    A = -100000
    for i in range(l):
        hMax = h[i]
        st = i
        en = i + 1
        while st >= 1 and h[st - 1] >= hMax:
            st -= 1
        while en <= l and h[en - 1] >= hMax:
            en += 1
        A = max(A,(en - 1 - st) * hMax)
    return A

# Street covering
def hackerlandRadioTransmitters(x, k):
    # input: cover street with lights of range k placing at positions in x
    # paradigm: Two-Pointers
    x.sort()
    st = x[0]
    en = x[0]
    tx = 1
    for house in x:
        if house > en + k:
            tx += 1
            en = house
            st = house
        else:
            if house <= st + k:
                en = house
    return tx

# Minimum of sub-sequences
def solve(arr, queries):
    # input: array and lengths of sub-sequences to check
    # paradigm: Two-Pointers
    ans = []
    for q in queries:
        i = 0
        j = q
        newMax = max(arr[i:j])
        tempMin = newMax
        while j < len(arr):
            if arr[i] == newMax:
                newMax = max(arr[i + 1:j + 1])
                if newMax < tempMin:
                    tempMin = newMax
            i += 1
            j += 1
        ans.append(tempMin)
    return ans

# ---------------------------- Divide and Conquer -------------------------

# bisecting binary search (left)
def binarySearch(l, k):
    # input: array and key
    # paradigm: Divide and Conquer
    left = bisect_left(l, k)
    if left != len(l) and l[left] == k:
        return left
    return -1

# bisecting binary search (right)
def rightBinarySearch(l, k):
    # input: array and key
    # paradigm: Divide and Conquer
    right = bisect_right(l, k)
    if right != 0 and l[right - 1] == k:
        return right - 1
    return -1

# Merge sort
def mergeSort(A, l, r):
    # input: array, left and right index
    # paradigm: Divide and Conquer
    def merge(A, l, m, r):
        # input: array, left, middle and right index
        n1 = m - l + 1
        n2 = r - m
        L, R = [], []
        for i in range(n1):
            L.append(A[l + i])
        for i in range(n2):
            R.append(A[m + 1 + i])
        i, j, k = 0, 0, l
        while i < n1 and j < n2:
            if L[i] <= R[j]:
                A[k] = L[i]
                i += 1
            else:
                A[k] = R[j]
                j += 1
            k += 1
        while i < n1:
            A[k] = L[i]
            i += 1
            k += 1
        while j < n2:
            A[k] = R[j]
            j += 1
            k += 1
    if l < r:
        m = l + (r - l) // 2
        mergeSort(A,l,m)
        mergeSort(A,m + 1,r)
        merge(A,l,m,r)

# ---------------------------- Dynamic Programming ---------------------------

# Coin change
def getWays(n, c):
    # input: change for n using coins in c
    # paradigm: Dynamic Programming
    ways = [0 for _ in range(n + 1)]
    ways[0] = 1
    for coin in sorted(c):
        for idx in range(n + 1):
            if coin <= idx:
                ways[idx] += ways[idx - coin]
    return ways[n]

# Largest common sub-array string
def commonChild(s1, s2):
    # input: string pair
    # paradigm: Dynamic Programming
    n = len(s2)
    dp = [0 for _ in range(n)]
    for c in s1:
        l = 0
        for i in range(n):
            if c == s2[i]:
                l, dp[i] = dp[i], l + 1
            else:
                l = max(l,dp[i])
    return max(dp)

# Community builder
def community(Q):
    # input: query to find or merge
    # paradigm: Dynamic Programming
    def helper(p1, p2, dp):
        # merge communities
        if p1 not in dp:
            dp[p1] = [p1]
        if p2 not in dp:
            dp[p2] = [p2]
        if dp[p1] is dp[p2]:
            return
        C1 = dp[p1]
        C2 = dp[p2]
        if len(C1) < len(C2):
            C1, C2 = C2, C1
        C1.extend(C2)
        for p in C2:
            dp[p] = C1
        return
    dp = {}
    for q in Q:
        if q[0] == 'Q':
            p = q[1]
            if p in dp:
                print(len(dp[p]))
            else:
                print(1)
        else:
            p1 = q[1]
            p2 = q[2]
            helper(p1,p2,dp)

# Maximum subset sum
def maxSubsetSum(arr):
    # input: array
    # paradigm: Dynamic Programming
    if max(arr) < 0:
        return 0
    dp = {0:arr[0]}
    dp[1] = max(arr[0],arr[1])
    maxSoFar = max(dp.values())
    for i, e in enumerate(arr):
        if i < 2:
            continue
        dp[i] = max(e,maxSoFar,dp[i - 2] + e)
        maxSoFar = max(maxSoFar,dp[i])
    return maxSoFar

##################################### Functions ###################################

# Decimal to binary conversion
def decToBin(n):
    # input: decimal integer
    # output: binary value
    if n == 0:
        return '0'
    s = ''
    d = 2 ** math.floor(math.log2(n))
    while n > 0 or d > 0:
        if n >= d:
            s += '1'
        else:
            s += '0'
        n %= d
        d //= 2
    return s

# Binary to decimal conversion
def binToDec(s):
    # input: binary string
    # output: decimal value
    n = 0
    if s == '0':
        return n
    for i, c in enumerate(s):
        if c == '1':
            n += 2 ** (i)
    return n

# First n primes
def getPrimes(n):
    # input: number of primes needed
    l = []
    d = 2
    while len(l) < n:
        for i in range(2,d):
            if d % i == 0:
                break
        else:
            l.append(d)
        d += 1
    return l

# Prefix sums
def prefixSums(l):
    # input: array
    sums = []
    currSum = 0
    for e in l:
        sums.append(currSum)
        currSum += e
    sums.append(currSum)
    return sums

# Suffix sums
def suffixSums(l):
    # input: array
    sums = []
    currSum = sum(l)
    for e in l:
        sums.append(currSum)
        currSum -= e
    sums.append(currSum)
    return sums

# Sort dictionary by value and internally by key (reversed)
def prioritySort(d):
    # input: dictionary
    return dict(sorted(d.items(), key = lambda x: (-x[1], x[0])), reverse = True)

# Triangle inequality
def isTriangle(x,y,z):
    # input: side lengths
    return (x + y > z) and (x + z > y) and (y + z > x)

# Counting sort
def countSort(arr):
    # input: 2-D array
    n = len(arr)
    ans = [[] for _ in range(n)]
    for i, l in enumerate(arr):
        k, v = l
        if i < n // 2:
            v = '-'
        ans[int(k)].append(v)
    return ' '.join([' '.join(i) for i in ans]).strip()

# Cumulative XOR
def listXOR(l):
    # input: list of elements
    def XOR (a,b = 0):
        return a ^ b
    n = len(l)
    i = 0
    ans = 0
    while i < n:
        ans = XOR(l[i],ans)
        i += 1
    return ans

# Substring list
def subStrings(s):
    # input: string
    # output: list of all sub-strings
    subs = []
    l = 1
    while l <= len(s):
        i = 0
        while i + l < len(s):
            subs.append(s[i:i + l])
            i += 1
        l += 1
    return subs

# Frequency computation
def freq(s):
    # input: string or list
    # output: dictionary of characters/values mapped to counts
    return Counter(s)