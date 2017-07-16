# codint: utf-8

from pprint import pprint

global a
a = [5, 2, 4, 7, 1, 3, 2, 6]

def merge(start, mid, end):
    global a
    n1 = mid - start + 1
    n2 = end - mid
    left = []
    right = []
    for i in range(n1):
        left.append(a[start+i])
    for j in range(n2):
        right.append(a[mid+1+j])
        
    k = start
    i = j = 0
    while i < n1 and j < n2:
        if left[i] < right[j]:
            a[k] = left[i]
            i += 1
            k += 1
        else:
            a[k] = right[j]
            j += 1
            k += 1
            
    while i < n1:
        a[k] = left[i]
        k += 1
        i += 1
    while j < n2:
        a[k] = right[j]
        k += 1
        j += 1

def sort(start, end):
    global a
    if start < end:
        mid = int((start + end) / 2)
        pprint("sort (%s - %s, %s - %s)" % (start, mid, mid+1, end))
        pprint(a)
        sort(start, mid)
        sort(mid+1, end)
        merge(start, mid, end)
        pprint("merge (%s - %s, %s - %s)" % (start, mid, mid+1, end))
        pprint(a)

sort(0, len(a)-1)
        
