#include <bits/stdc++.h>
#include<stdio.h>
using namespace std;
signed main()
{
int test,t;
cin>>test;
for(t=0;t<test;t++)
{
int l,i,j,trace=0,r=0,c=0;
map<pair<int,int>,int>k1;
map<pair<int,int>,int>k2;
cin>>l;
int f[l][l];
for(i=0;i<l;i++)
{
for(j=0;j<l;j++)
{
cin>>f[i][j];
if(i==j)
trace+=f[i][j];
}
}
int f1=0;
int f2[l]={0};
for(i=0;i<l;i++)
{
f1=0;
for(j=0;j<l;j++)
{
if(k1[{f[i][j],i}]==1&& f1==0)
{
r++;
f1=1;
}
else
k1[{f[i][j],i}]=1;
if(k2[{f[i][j],j}]==1 && f2[j]==0)
{
c++;
f2[j]=1;
}
else
k2[{f[i][j],j}]=1;
}
}
cout<<"Case #"<<t+1<<": "<<trace<<" "<<r<<" "<<c<<endl;
}
return 0;
}
