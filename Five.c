#include<stdio.h>
int result[10][10], Size, trace, test;
int p[10][10], q[10][10], flag=0;
void natural_latin(int r,int c,int trace1);
int main()
 {
    int totaltestcases;
    scanf(" %d", &totaltestcases);
    for (test = 1; test <= totaltestcases; test++)
    {
        scanf(" %d %d",&Size,&trace);
        natural_latin(1, 1, 0);
        if (!flag)
        {
             printf("Case #%d: IMPOSSIBLE\n",test);
        }
        flag = 0;
    }
    return 0;
}
void natural_latin(int r,int c,int trace1) {
    if (r==Size && c==Size+1 && trace1==trace && !flag)
       {
        flag = 1;
        printf("Case #%d: POSSIBLE\n",test);
        for (int i = 1; i <= Size;i++) {
            for (int j = 1; j <= Size; j++) {
                printf("%d ",result[i][j]);
            }
            printf("\n");
        }
        return;
    }
    else if (r>Size)
    {
        return;
    }
    else if (c>Size)
    {
        natural_latin(r+1,1,trace1);
    }
    for (int i = 1; i <= Size && !flag; i++)
    {
          if (!p[r][i] && !q[c][i])
            {
             p[r][i] = q[c][i] = 1;
             if(r==c)
            {
                trace1 = trace1 + i;
            }
            result[r][c] = i;
            natural_latin(r,c+1,trace1);
            p[r][i] = q[c][i] = 0;
            if (r == c)
            {
                trace1 = trace1 - i;
            }
            result[r][c] = 0;
        }
    }
}
