# ------------------------------ Doubly Linked List ----------------------------

# node class
class DoublyLinkedListNode:
    
    # initialise
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None
    
    # node printing
    def __str__(self):
        return str(self.data) 
    
# list class
class DoublyLinkedList:
    
    # initialise
    def __init__(self):
        self.head = None
        self.tail = None

    # insert value
    def insert(self, data):
        # input: data to be inserted
        node = DoublyLinkedListNode(data) # new node
        if not self.head: # no head
            self.head = node
        else:
            self.tail.next = node # add behind tail
            node.prev = self.tail
        self.tail = node # move tail
    
    # print the list
    def printer(self, sep = ' '):
        # input: separator to use
        ptr = self.head
        while ptr != None:
            print(ptr.data, end = '')
            ptr = ptr.next
            if ptr != None:
                print(sep)
    
    # reverse the list
    def reverse(self):
        head = self.head # head pointer
        prev = None # previous pointer
        while head != None: # new node left
            newHead = head.next # save pointer to next node (cut forward link)
            if newHead != None: # check if next node has a reverse link
                newHead.prev = head # save pointer to previous node (cut reverse link)
            head.next = prev # reverse the forward link
            head.prev = newHead # reverse the reverse link
            prev = head # move pointer to previous element
            head = newHead # use saved pointer to move head
        self.tail = self.head
        self.head = prev
    
    # sorted insertion
    def sortedInsert(self, data):
        # input: data to be added
        head = self.head
        if head is None:
            head.data = data
        else:
            ptr1 = ptr2 = head
            while ptr2 != None and ptr2.data < data:
                ptr1 = ptr2
                ptr2 = ptr2.next
            node = DoublyLinkedListNode(data)
            node.next = ptr2
            if ptr1 != ptr2:
                ptr1.next = node
                node.prev = ptr1
                if ptr2 != None:
                    ptr2.prev = node
            else:
                ptr2.prev = node
        while head.prev != None:
            head = head.prev
        self.head = head
        while head.next != None:
            head = head.next
        self.tail = head