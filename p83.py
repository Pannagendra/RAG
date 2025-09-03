"""
You are given the head of a doubly linked list. You have to reverse the doubly linked list and return its head.
Input:3 <-> 4 <-> 5
   
Output: 5 <-> 4 <-> 3
Explanation: After reversing the given doubly linked list the new list will be 5 <-> 4 <-> 3.
   
"""

class Solution:
    def reverse(self, head):
        if not head:
            return None
        
        curr = head
        new_head = None
        
        while curr:
            # swap next and prev
            curr.prev, curr.next = curr.next, curr.prev
            
            # update new_head before moving
            new_head = curr  
            
            # move to the next node in the *original* list
            curr = curr.prev  # (since we swapped, prev is actually the original next)
        
        return new_head
