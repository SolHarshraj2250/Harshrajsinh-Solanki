#include <bits/stdc++.h>
using namespace std;
int main()
{
int test,a;
cin>>test;
for(a=0;a<test;a++)
{
string l;
int i,n;
cin>>n;
vector<pair<int,pair<int,int> > >ve;
int p[n],k[n];
for(i=0;i<n;i++)
{
cin>>p[i]>>k[i];
ve.push_back({p[i],{k[i],i}});
l+='C';
}
sort((ve).begin(),(ve).end());
int c=0,j=0,f=0;
for(i=0;i<ve.size();i++)
{
if(ve[i].first>=c)
{
l[ve[i].second.second]='C';
c=ve[i].second.first;
}
else if(ve[i].first>=j)
{
l[ve[i].second.second]='J';
j=ve[i].second.first;
}
else
{
f=1;
break;
}
}
if(f==1)
{
cout<<"Case #"<<a+1<<": "<<"IMPOSSIBLE"<<endl;
continue;
}
cout<<"Case #"<<a+1<<": "<<l<<endl;
}
return 0;
}
