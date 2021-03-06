#include <stdio.h>

#define LEN  8

int a[LEN] = {13, 4, 212, 32, 53, 111, 44, 3};

int partition(int start, int end){
	int pivot;
	int mid;
	int left[LEN], right[LEN];
	int i;
	int l_idx, r_idx;

	pivot = a[start];
	l_idx = r_idx = -1;
	for (i=start+1; i<=end; i++){
		if (a[i] < pivot)
		       left[++l_idx] = a[i];
		else
			right[++r_idx] = a[i];
	}
	
	for (i=0; i<=l_idx; i++)
		a[start+i] = left[i];
	mid = start + l_idx + 1;
	a[mid] = pivot;

	for (i=0; i<=r_idx; i++)
		a[mid+i+1] = right[i];
	
	for (i=0; i<LEN; i++)
		printf("%d ", a[i]);
	printf("\n");
	return mid;
}

void quicksort(int start, int end){
	int mid, i;
	if (end > start){
		mid = partition(start, end);
		quicksort(start, mid-1);
		quicksort(mid+1, end);
	}
}

int main(void){
	quicksort(0, LEN-1);
	return 0;
}
