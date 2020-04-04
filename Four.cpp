#include <bits/stdc++.h>
using namespace std;
int main()
{
	int test,bits;
	cin>>test>>bits;
	while(test--)
	{
		int i;
	   	char t,t1;
	    string p="";
		for(i=0;i<10;i++)
		p=p+'0';
		for(i=1;i<11;i++)
		{
			cout<<i<<endl;
			cout.flush();
			cin>>t;
			p[i-1]=t;
		}
		cout<<p<<endl;
		cout.flush();
		cin>>t1;
		if(t1=='Y')
		continue;
		else
		return 0;
}
	return 0;
}
