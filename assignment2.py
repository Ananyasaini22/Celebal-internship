class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def print_list(self):
        if self.head is None:
            print("List is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        if self.head is None:
            print("Cannot delete from an empty list.")
            return
        if n <= 0:
            print("Invalid index. Index should be 1 or greater.")
            return
        if n == 1:
            self.head = self.head.next
            return
        current = self.head
        count = 1
        while current and count < n - 1:
            current = current.next
            count += 1
        if current is None or current.next is None:
            print("Index out of range.")
            return
        current.next = current.next.next

my_list = LinkedList()
my_list.add_node(10)
my_list.add_node(20)
my_list.add_node(30)
my_list.add_node(40)

print("Original List:")
my_list.print_list()

print("\nDeleting 2nd node:")
my_list.delete_nth_node(2)
my_list.print_list()

print("\nTrying to delete 10th node:")
my_list.delete_nth_node(10)

print("\nDeleting all nodes:")
my_list.delete_nth_node(1)
my_list.delete_nth_node(1)
my_list.delete_nth_node(1)
my_list.print_list()

print("\nTrying to delete from empty list:")
my_list.delete_nth_node(1)