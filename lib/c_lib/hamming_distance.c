#include <stdio.h>
#include <string.h>

int hamming_distance(const char *str1, const char *str2)
{

    int strlen_1 = strlen(str1), strlen_2 = strlen(str2);

    int i = 0, min = 0, distance = 0;

    if (strlen_1 < strlen_2)
    {
        min = strlen_1;
    }
    else if (strlen_2 < strlen_1)
    {
        min = strlen_2;
    }

    for (i = 0; i < min; ++i)
    {
        if (str1[i] != str2[i])
        {
            distance++;
        }
    }

    return distance;
}