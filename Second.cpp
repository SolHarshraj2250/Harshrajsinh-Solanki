#include <bits/stdc++.h>
using namespace std;
int main()
{
int test,m;
cin>>test;
for(m=0;m<test;m++)
{
int i;
string s1,result;
cin>>s1;
int cou=0;
for(i=0;i<s1.length();i++)
{
int t=(int)s1[i]-48;
if(t<cou)
{
while(t!=cou)
{
result=result+')';
cou--;
}
cou=t;
}
else if(t>cou)
{
while(t!=cou)
{
result=result+'(';
cou++;
}
cou=t;
}
result=result+s1[i];
}
while(cou>0)
{
result=result+')';
cou--;
}
cout<<"Case #"<<m+1<<": "<<result<<endl;
}
return 0;
}
