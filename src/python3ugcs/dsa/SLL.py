# ------------------------------- Singly Linked List -----------------------------

# node class
class SinglyLinkedListNode:
    
    # initialise
    def __init__(self, data):
        self.data = data
        self.next = None
    
    # node printing
    def __str__(self):
        return str(self.data)

# list class
class SinglyLinkedList:
    
    # initialise
    def __init__(self):
        self.head = None
        self.tail = None
    
    # insert value
    def insert(self, data):
        # input: data to be inserted
        node = SinglyLinkedListNode(data) # new node
        if not self.head: # no head
            self.head = node
        else:
            self.tail.next = node # add behind tail
        self.tail = node # move tail
    
    # insert at position
    def insertAtPos(self, data, position):
        # input: data to be inserted and position to insert
        node = SinglyLinkedListNode(data) # new node
        head = self.head
        while position > 1: # move to position just before insertion
            head = head.next
            position -= 1
        ptr = head.next # save pointer to next node (cut link)
        head.next = node # add new node to list
        node.next = ptr # add back remaining list
    
    # find value
    def find(self, data):
        # input: data to be found
        head = self.head
        prev = None
        while head != None and head.data != data:
            prev = head
            head = head.next
        return prev
    
    # delete value
    def delete(self, data):
        # input: data to be deleted
        prevPos = self.find(data)
        if prevPos.next == None:
            return False
        prevPos.next.next = prevPos.next
        return True
        
    # print the list
    def printer(self, sep = ' '):
        # input: separator to use
        ptr = self.head
        while ptr != None:
            print(ptr.data, end = '')
            ptr = ptr.next
            if ptr != None:
                print(sep, end = '')
        print()
    
    # reverse the list
    def reverse(self):
        head = self.head # head pointer
        prev = None # previous pointer
        while head != None: # while there is forward link left
            newHead = head.next # save extra pointer to next element
            head.next = prev # reverse the link of current element
            prev = head # move pointer to previous element
            head = newHead # use extra pointer to move to next element
        self.tail = self.head
        self.head = prev
    
    # cycle detection
    def hasCycle(self):
        head = self.head
        seenPtrs = []
        while head != None:
            if head in seenPtrs:
                return True
            seenPtrs.append(head)
            head = head.next
        return False

# find merge point
def findMergeNode(head1, head2):
    # input: head of two lists
    while head1 != None:
        ptr = head2
        while ptr != None:
            if head1 == ptr:
                return head1
            ptr = ptr.next
        head1 = head1.next

# merged sorted linked lists
def mergeSLL(head1, head2):
    # input: head of two lists
    merged = SinglyLinkedList()
    while head1 != None and head2 != None: # both lists not empty
        if head1.data < head2.data: # link node with smaller data
            merged.insert(head1.data)
            head1 = head1.next
        else:
            merged.insert(head2.data)
            head2 = head2.next
    if head1 == None and head2 != None: # list 1 finished
        merged.tail.next = head2 # add remaining list 2 as is
    if head1 != None and head2 == None: # list 2 finished
        merged.tail.next = head1 # add remaining list 1 as is
    return merged.head