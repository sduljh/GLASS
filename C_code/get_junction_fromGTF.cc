#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<algorithm>
#include<vector>
#include<map>

using namespace std;
void load_ref(char* file, map<string, map<pair<int,int>,bool> >& Chr_Junc_map)
//map<string, map< vector<int>,string> >& Chr_vecExon_map,map<string, vector<pair<int,int> > >& Chr_single_map)
{
    ifstream in(file);
    istringstream istr;
    string s;

    string chr, strand, lable;
    string tranid;
    string temp;
    int exon_l,exon_r;
    vector<int> vecExon;

    getline(in,s);
    istr.str(s);

    istr>>chr>>temp>>lable>>exon_l>>exon_r>>temp>>strand;
    while(istr>>temp) if( temp == "transcript_id") istr>>tranid;
    if(lable == "exon"){ vecExon.push_back(exon_l); vecExon.push_back(exon_r);}
    istr.clear();


    while(getline(in,s))
    {
	istr.str(s);
 	string current_id;
	string curr_chr, curr_strand;
	int curr_el, curr_er;

	istr>>curr_chr>>temp>>lable>>curr_el>>curr_er>>temp>>curr_strand;
	if(lable != "exon") continue;

	while(istr>>temp) if( temp == "transcript_id") istr>>current_id;
	istr.clear();
	if(current_id == tranid)//commom transcript
	{
	    vecExon.push_back(curr_el); vecExon.push_back(curr_er);
	}
	else 
	{
	    //if(vecExon.size() == 2) continue;
	    if(vecExon.size() != 2)
	    {
	      sort(vecExon.begin(),vecExon.end());
	      vecExon.erase(vecExon.begin()); vecExon.pop_back();
	      //for(int i=0;i<vecExon.size();i++) cerr<<vecExon[i]<<" ";
	      //cerr<<endl;
	      for(size_t i=0;i<vecExon.size();)
	      {
		cout<<strand<<" "<<chr<<" "<<vecExon[i]<<" "<<vecExon[i+1]<<endl;
	  	i += 2;
	      }
	      chr += strand;
	      if(Chr_Junc_map.find(chr) == Chr_Junc_map.end())
	      {
		map<pair<int,int>,bool> m;
		for(size_t i=0;i<vecExon.size();)
	 	{
		    pair<int,int> junc = make_pair(vecExon[i],vecExon[i+1]);
		    m[junc] = true;
		    i += 2;
		}
		Chr_Junc_map[chr] = m;
	      } else {
		for(size_t i=0;i<vecExon.size();)
		{
		    pair<int,int> junc = make_pair(vecExon[i],vecExon[i+1]);
		    Chr_Junc_map[chr][junc] = true;
		    i += 2;
		}
	      }
	    }
	    else {
	    }
	    vecExon.clear();
	    vecExon.push_back(curr_el); vecExon.push_back(curr_er);
	    chr = curr_chr; strand = curr_strand;
	    tranid = current_id;
	    
	}
    }

    if(vecExon.size() != 2) {
     vecExon.erase(vecExon.begin()); vecExon.pop_back();
     sort(vecExon.begin(),vecExon.end());
     //for(int i=0;i<vecExon.size();i++) cerr<<vecExon[i]<<" ";
     //cerr<<endl;
     for(size_t i=0;i<vecExon.size();)
     {
	cout<<strand<<" "<<chr<<" "<<vecExon[i]<<" "<<vecExon[i+1]<<endl;
	i += 2;	
     }
     chr += strand;
	      if(Chr_Junc_map.find(chr) == Chr_Junc_map.end())
              {
                map<pair<int,int>,bool> m;
                for(size_t i=0;i<vecExon.size();)
                {
                    pair<int,int> junc = make_pair(vecExon[i],vecExon[i+1]);
                    m[junc] = true;
                    i += 2;
                }
                Chr_Junc_map[chr] = m;
              } else {
                for(size_t i=0;i<vecExon.size();)
                {
                    pair<int,int> junc = make_pair(vecExon[i],vecExon[i+1]);
                    Chr_Junc_map[chr][junc] = true;
                    i += 2;
                }
              }
    }
    return;
}
 
void load_and_check_ass(char* file,map<string, map<pair<int,int>,bool> >& Chr_Junc_map)
//map<string, map< vector<int>,string > >& Chr_vecExon_map,map<string, vector<pair<int,int> > >& Chr_single_map)
{
    int single = 0;
    ifstream in(file);
    istringstream istr;
    string s;

    string chr, strand, lable;
    string tranid;
    string temp;
    int exon_l,exon_r;
    vector<int> vecExon;

    //getline(in,s);
    //getline(in,s);
    getline(in,s);
    istr.str(s);

    istr>>chr>>temp>>lable>>exon_l>>exon_r>>temp>>strand;
    while(istr>>temp) if( temp == "transcript_id") istr>>tranid;
    if(lable == "exon"){ vecExon.push_back(exon_l); vecExon.push_back(exon_r);}
    istr.clear();


    while(getline(in,s))
    {
        istr.str(s);
        string current_id;
        string curr_chr, curr_strand;
        int curr_el, curr_er;

        istr>>curr_chr>>temp>>lable>>curr_el>>curr_er>>temp>>curr_strand;
        if(lable != "exon") continue;

        while(istr>>temp) if( temp == "transcript_id") istr>>current_id;
	istr.clear();
        if(current_id == tranid)//commom transcript
        {
            vecExon.push_back(curr_el); vecExon.push_back(curr_er);
        }
	else 
	{
	    for(int i = 1;i<vecExon.size() - 1;)
	    {
		if(vecExon[i+1] - vecExon[i] ==1)
		{
		    vecExon.erase(vecExon.begin() + i);
		    vecExon.erase(vecExon.begin() + i);
		}
		else i += 2;
	    }
	    if(vecExon.size() == 2) single++;
	    if(vecExon.size() != 2){
	      vecExon.erase(vecExon.begin()); vecExon.pop_back();
              sort(vecExon.begin(),vecExon.end());

	      //for(int i=0;i<vecExon.size();i++) cerr<<vecExon[i]<<" ";
	      //cerr<<endl;

	      chr += strand;
	
	      map<pair<int,int>,bool> m = Chr_Junc_map[chr];
	      bool flag = false;
	      for(size_t i=0;i<vecExon.size();)
	      {
		pair<int,int> junc = make_pair(vecExon[i],vecExon[i+1]);
		if( !m[junc] )
		{
		    cerr<<tranid<<" "<<junc.first<<" "<<junc.second<<endl;
		    break;
		}
		i=i+2;
 	      }
	    }
	    else {
	    }
	    vecExon.clear();
            vecExon.push_back(curr_el); vecExon.push_back(curr_er);
            chr = curr_chr; strand = curr_strand;
            tranid = current_id;
	}
    }
    
    for(int i = 1;i<vecExon.size() - 1;)
    {
        if(vecExon[i+1] - vecExon[i] ==1)
        {
                vecExon.erase(vecExon.begin() + i);
                vecExon.erase(vecExon.begin() + i);
         }
         else i += 2;
    }
    if(vecExon.size() == 2) single++;
    {
	vecExon.erase(vecExon.begin()); vecExon.pop_back();
	sort(vecExon.begin(),vecExon.end());
	chr += strand;
	map<pair<int,int>,bool> m = Chr_Junc_map[chr];
	for(size_t i=0;i<vecExon.size();)
	{
	    pair<int,int> junc = make_pair(vecExon[i],vecExon[i+1]);
	    if( !m[junc] )
	    {

	    	cerr<<tranid<<" "<<junc.first<<" "<<junc.second<<endl;
		break;
	    }
	    i+=2;
	}
    }
cerr<<"single exon number: "<<single<<endl;
    return;
}

//./exon SC.gtf ipac_right.gtf
//./exe UCSC-expressed.gtf > Junction(- chr 12223 13345)
int main(int argc,char* argv[])
{
    if(argc==1)
    {
      cout<<"*"<<endl;
      cout<<"This is a simple program to get junction from a GTF file"<<endl;
      cout<<"./get_junction_fromGTF file.gtf >junction.info"<<endl;
      cout<<"*"<<endl;
      return 0;
    }
    map<string, map<pair<int,int>,bool> >Chr_Junc_map;
    typedef map<string, map<pair<int,int>,bool> >::iterator iter;
    load_ref(argv[1], Chr_Junc_map);
/*
 */   
    //load_and_check_ass(argv[2],Chr_Junc_map);
}
