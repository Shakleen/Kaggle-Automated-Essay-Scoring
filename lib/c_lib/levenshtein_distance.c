#include <stdio.h>
#include <string.h>

#define MAX_STR_LEN 10000

int distances[MAX_STR_LEN][MAX_STR_LEN];

void initDistanceArray(int strlen_1, int strlen_2)
{
    int i = 0, j = 0;

    for (i = 0; i <= strlen_1; i++)
    {
        for (j = 0; j <= strlen_2; j++)
        {
            distances[i][j] = 0;
        }
    }

    for (i = 0; i <= strlen_1; i++)
    {
        distances[i][0] = i;
    }

    for (i = 0; i <= strlen_2; i++)
    {
        distances[0][i] = i;
    }
}

int levenshtein_distance(const char *str1, const char *str2)
{

    int strlen_1 = strlen(str1), strlen_2 = strlen(str2);

    initDistanceArray(strlen_1, strlen_2);

    int i = 0, j = 0, a = 0, b = 0, c = 0;
    for (i = 1; i <= strlen_1; i++)
    {
        for (j = 1; j <= strlen_2; j++)
        {
            if (str1[i - 1] == str2[j - 1])
            {
                distances[i][j] = distances[i - 1][j - 1];
            }
            else
            {
                a = distances[i][j - 1];
                b = distances[i - 1][j];
                c = distances[i - 1][j - 1];

                if (a <= b && a <= c)
                {
                    distances[i][j] = a + 1;
                }
                else if (b <= a && b <= c)
                {
                    distances[i][j] = b + 1;
                }
                else
                {
                    distances[i][j] = c + 1;
                }
            }
        }
    }

    // for (i = 0; i <= strlen_1; i++)
    // {
    //     for (j = 0; j <= strlen_2; j++)
    //     {
    //         printf("%d ", distances[i][j]);
    //     }

    //     printf("\n");
    // }

    return distances[strlen_1][strlen_2];
}