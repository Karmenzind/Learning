#include <stdio.h>

#define LEN 5

int a[] = {10, 5, 2, 4, 7};

void selection(void){
	int i, j;
	int idx, val;
	for (i=0; i<LEN; i++){
		idx = i;
		val = a[i];
		for (j=i+1; j<LEN; j++){
			if (a[j] < val){
				idx = j;
				val = a[j];
			}
		}
		
		if (a[i] != val){
			a[idx] = a[i];
			a[i] = val;
		}
		for (int m=0; m<LEN; m++)
			printf("%d\t", a[m]);
		printf("\n");

	}
}

int main(void)
{
	selection();
	return 0;
}
