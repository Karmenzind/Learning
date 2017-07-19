#include <stdio.h>

char a[] = "hello world";

int indexof(char letter)
{
	int i = 0;
	while (a[i] != '\0') {
		if (a[i] == letter)
			return i;
		i++;
	}
	return -1;
}

int main(void)
{
	printf("%d %d %d\n", indexof('w'), indexof('o'), indexof('a'));
	return -1;
}
