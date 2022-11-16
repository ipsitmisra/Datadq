from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd
from fuzzywuzzy import fuzz, process
import csv
from email_validator import validate_email, EmailNotValidError
from pyzipcode import ZipCodeDatabase
from pandas import *
from pyzipcode import ZipCodeDatabase
import re
import numbers
import decimal
import string
import collections as ct
from commonregex import CommonRegex
from validate_email import validate_email
from sklearn.ensemble import IsolationForest
import zipfile
import os
from collections import Counter 
from statistics import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import io




#uploading zip file,storing it
def multifile(request):
    if request.method == 'POST' :
        myfile = request.FILES['file1']
        fs = FileSystemStorage()
        filename=fs.save(myfile.name, myfile)
        #print(myfile)
        files=zipextract(myfile)
        #print(files)
        #out = {'files' : files,}
        commonmulti = common_table(files)
        return render( request, 'multidataset.html')
    return render( request, 'multidataset.html' )

#extracting zip file and returning a list of it
def zipextract(file_path):
    text = ""
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall("media\\zip")
        file3 = os.listdir("media\\zip")
        f1=[]
        for f in file3:
            f1.append(f)
        return(f1)

def final(request):
    return render( request, 'main.html')
    

#for single dataset this is the main function
def home(request):
    if request.method == 'POST' :
        
        myfile = request.FILES.get('file1', False)
        if myfile==False:
            pass
      
        else:
            fs = FileSystemStorage()
            filename=fs.save(myfile.name, myfile)
    
            #ext=myfile.split(".")[1]
            #exten="."+ext
            fname="P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\media\\"+myfile.name
            data22=fname,str(".csv")
            print(fname)
            data = pd.read_csv(fname)
            data1 = formatting(data)
            Pattern, cols= PatternRecog(data, data1)
            Pattern.columns = cols
            phone_no, not_phone_no = operations_phone(data, Pattern)
            #mail_id, not_mail_id = operations_mail(data, Pattern)
            zip_code, not_zip_code = operations_zip(data, Pattern)
            mail_id, not_mail_id = operations_mail(data, Pattern)
            address, not_address = operations_address(data, Pattern)
        return render( request, 'binod.html' )
    return render( request, 'binod.html' )
    

    return render(request,'binod.html')
    
#rendering all hrml files   
def analyze(request):
    return render( request, 'analysis.html')

def ph1(request):
    return render( request, 'ph1.html')
    
def phanomoly1(request):
    return render( request, 'phanomoly1.html')

def zip1(request):
    
    return render( request, 'zip1.html')
    
def zipanomoly1(request):
    return render( request, 'zipanomoly1.html')
    
def mail1(request):
    return render( request, 'mail1.html')

def mailanomoly1(request):
    return render( request, 'mail_anomoly1.html')
    
def address1(request):
    return render( request, 'address1.html')
    
def addressanomoly1(request):
    return render( request, 'addressanomoly1.html')   
    
def common(request):
    return render( request, 'mergetables.html')      

def unique(request):
    return render( request, 'unique_elements_combined.html')      

def uniquein1(request):
    return render( request, 'unique_elements_df241.html')      

def uniquein2(request):
    return render( request, 'unique_elements_df242.html')          
   
def coloutlier(request):
    return render( request, 'outliers.html')

def dataoutlier(request):
    return render( request, 'totaldataoutlier.html')

def colorout(request):
    return render( request, 'color.html')
    
    
#for the multitable 
# All possible analysis between two dataset
def common_table(files):  
    df24=[]
    #size = []
    for i in files:
        path="P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\media\\zip\\"+i
        #print(path)
        df_i = pd.read_csv(path)
        #size.append(df_i.shape[0])
        df24.append(df_i)
    #concat a fuction
    df241 = df24[0]
    df242 = df24[1]
    #merge Element
    if set(df241.columns).intersection(set(df242.columns)):
        df25 = df241.merge(df242,how='outer')
    else:
        df25 = pd.concat([df241,df242])
        
    colorut = encoder(df25) 
    df53 = df25[column].style.apply(highlight_outlier, axis = 0)
    df53.to_excel('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\static\\a.xlsx', index = False)

    # Conversion of the df_highlight dataframe to html file

    html = df53.hide_index().render()
    with open("P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\static\\color.html","w") as fp:
      fp.write(html)
    
    df25.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\mergetables.html')
    df25.to_csv(r'P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\static\\merge.csv', index = False)
    
    
    #unique elements in df1 and df2 i.e., df1 + df2 - intersected_df 
    unique_elements_combined = pd.concat([df241,df242]).drop_duplicates(keep=False)
    
    #df1 - df2
    unique_elements_df241 = df241.merge(df242, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge',axis='columns')
    
    #df2 - df1
    unique_elements_df242 = df242.merge(df241, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge',axis='columns')
    
    unique_elements_combined.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\unique_elements_combined.html')
    unique_elements_df241.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\unique_elements_df241.html')
    unique_elements_df242.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\unique_elements_df242.html')
    
    #analyzing the data and anomaly detecting
    data2 = formatting(df25)
    Pattern, cols= PatternRecog(df25, data2)
    Pattern.columns = cols
    phone_no, not_phone_no = operations_phone(df25, Pattern)
    zip_code, not_zip_code = operations_zip(df25, Pattern)
    mail_id, not_mail_id = operations_mail(df25, Pattern)
    address, not_address = operations_address(df25, Pattern)
    #plotting 
    plot = make_plots(df25,"mergeplot")
    #total outlier
    total = totaloutlier(df25)
    #column outliers
    col, dat, stats = operations_outlier(df25)
    df1 = pd.DataFrame([dat])
    df1 = df1.transpose()
    df = pd.DataFrame(stats)
    # col gives columns with outliers
    # dat gives the data of outliers found
    # stats gives [mean, median, mode] in list form of all the outliers
    df1.index = col
    df.index = col
    df.index.name, df1.index.name = 'column', 'column'
    df.columns = ['mean', 'median', 'mode']
    df1.columns = ['outlier']
    df2 = pd.merge(df, df1, on = ['column'])
    #outlier with stats
    df2.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\outliers.html')

def encoder(df_in):

  global column
  column = []
  cols_dtype = [i for i in df_in.dtypes]
  for i,j in zip(df_in.columns, cols_dtype):
    if j !='object':
      column.append(i)

# To know which column has the outlier using z score method

  z = np.abs(stats.zscore(df_in[column]))

  threshold = 2.5
  rows_cols = np.where(z > 2.5)

  df_zscore = pd.DataFrame(rows_cols, index = ['Rows', 'Columns'])

# Anomaly dataframe contains the outlier detected using z score

  anomaly=df_in[column].iloc[rows_cols]
  df_anomaly = pd.DataFrame(anomaly)

# Function to print rows and columns values which contains the outlier

  final_df = df_zscore.transpose()

  rows = list(final_df['Rows'])
  cols = list(final_df['Columns'])

# Details about both rows and columns containing outliers
  global final
  final = []
  for i in range(len(cols)):
    val = df_in[column[cols[i]]][rows[i]]
    final.append(val)
 
def highlight_outlier(s):
  l = []
  for i in s:
    if i in final:
      l.append(True)
    else:
      l.append(False)
  vals = pd.Series(l)
  return ['color: red' if v else '' for v in vals]
  


  

  
#columnwise outliers 
# taking inter quartile range
def iqr(data):
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1
  low_lim = Q1 - 1.5 * IQR 
  up_lim = Q3 + 1.5 * IQR
  return low_lim, up_lim

# function for finding mode
def most_frequent(List): 
	occurence_count = Counter(List) 
	return occurence_count.most_common(1)[0][0] 

# finding mean median mode	
def mmm(data, col):
  daa = []
  for i in col:
      me = mean(data[i])
      med = median(data[i])
      if len(data[i])-len(data[i].drop_duplicates())!=0:
        mod = most_frequent(data[i])
      else:
        mod = 'Nan'
      daa.append([me, med, mod])
  return daa

# finding outlier
def operations_outlier(data):
  cols = [i for i in data.columns]
  column = []
  cols_dtype = [i for i in data.dtypes]
  for i,j in zip(cols, cols_dtype):
    if j!='object':
      column.append(i)
  data[column] = data[column].fillna(0)
  col_name, col_outlier = [],  []
  for i in column:
    low, high = iqr(data[i])
    outlier = [] 
    for x in data[i]: 
      if ((x>= high) or (x<= low)): 
          outlier.append(x) 
    i_out = []
    i_out = list(set(outlier))
    col_name.append(i)
    col_outlier.append(i_out)
    mem = mmm(data[col_name], col_name)
  return(col_name, col_outlier, mem)
    
#plotting the graph
def patches(plot, total=None, normalize=True, precision=2):
    sizes = []
    for p in plot.patches:
        height = p.get_height()
        sizes.append(height)
        if normalize:
            plot.text(p.get_x()+p.get_width()/2, height, round(height /
                                                               total*100, precision), ha="center", fontsize=12)
        else:
            plot.text(p.get_x()+p.get_width()/2, height*1.01 +
                      10, int(height), ha="center", fontsize=12)
    plot.set_ylim(0, max(sizes)*1.1)



def make_plots(data, filename):
    # uniques and duplicates
    total_length = data.shape[0]
    uniques = data.drop_duplicates().shape[0]
    duplicates = total_length - uniques
    outliers = totaloutlier(data).shape[0]
    plot_data = pd.DataFrame({'stats': ['uniques', 'duplicates', 'outliers','total'], 'values': [
        uniques, duplicates, outliers, total_length]})
    plt.figure(figsize=(10, 5))
    g = sns.barplot(x='stats', y='values', data=plot_data)
    patches(g, total_length, normalize=False, precision=0)
    g.set_xlabel('')
    g.set_ylabel('Count', fontsize=12)
    #plt.show()
    plt.savefig(f'P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\static\\img\\duplicates_{filename}.jpg')

#total dataset outlier
def totaloutlier(data):
    column = []
    cols_dtype = [i for i in data.dtypes]
    for i,j in zip(data.columns, cols_dtype):
      if j!='object':
        column.append(i)
    data[column] = data[column].fillna(0)

    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.05),max_features=1.0)
    model.fit(data[column])

    data['scores']=model.decision_function(data[column])
    data['outliers']=model.predict(data[column])

    anomaly=data.loc[data['outliers']==-1]
    anomaly_index=list(anomaly.index)
    #print(anomaly)
    df2 = DataFrame (anomaly)
    print(df2.shape)
    print(anomaly.shape)
    df2.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\totaldataoutlier.html')
    df2.to_csv(r'P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\totaldataoutlier.csv', index = False)
    return(anomaly)
    
#generating profiling reports, dataset as parameter
#using the segment
def formatting(Datadf):
    data = pd.DataFrame()
    data['Attributes'] = Datadf.columns
    df = pd.DataFrame()
    df = Datadf.describe(include='all')
    df = df.transpose()
    dc = df['count']
    du = Datadf.nunique().to_list()
    dfr = df['freq']
    data['Count'] = 0
    data['Unique'] = 0
    data['Frequency'] = 0
    data['DataTypes'] = 0
    data['IsNull'] = 0
    data['IsNotNull'] = 0
    data['PercentageNull'] = 0

    for i in range(len(Datadf.columns)):
        data['Count'][i] = dc[i]
        data['Unique'][i] = du[i]
        data['Frequency'][i] = dfr[i]
    ddt = Datadf.dtypes
    disn = Datadf.isnull().sum()
    disnn = Datadf.shape[0] - Datadf.isnull().sum()
    dpnl = (Datadf.isnull().sum()/Datadf.shape[0]) * 100
    for i in range(len(Datadf.columns)):
        data['DataTypes'][i] = ddt[i]
        data['IsNull'][i] = disn[i]
        data['IsNotNull'][i] = disnn[i]

    count = 0
    count1 = 0
    count2 = 0
    c = []
    c1 = []
    c2 = []
    data['Entire_Upper'] = 0
    data['Entire_Lower'] = 0
    for column in Datadf.columns:
        for i in range(len(Datadf)):
            if(str(Datadf[column][i]).isupper()):
                count = count+1
            if(str(Datadf[column][i]).islower()):
                count1 = count1+1
        c.append(count)
        c1.append(count1)
        count = 0
        count1 = 0

    data['Entire_Upper'] = c
    data['Entire_Lower'] = c1

    data['WithSpaces'] = 0
    data['WithoutSpaces'] = 0

    for column in Datadf.columns:
        for i in range(len(Datadf)):
            a = str(Datadf[column][i])
            t = a.split(" ")
            if len(t) > 1:
                count2 = count2+1
        c2.append(count2)
        count2 = 0

    data['WithSpaces'] = c2
    data['WithoutSpaces'] = Datadf.shape[0]-data['WithSpaces']

    data['Duplicates'] = data['Count'] - data['Unique']

    data['Min Words'] = 0
    data['Max Words'] = 0

    max = 0
    min = 9999
    c3 = []
    c4 = []
    for column in Datadf.columns:
        for i in range(len(Datadf)):
            a = str(Datadf[column][i])
            t = a.split(" ")
            if(len(t) > max):
                max = len(t)
            if(len(t) < min):
                min = len(t)
        c3.append(min)
        c4.append(max)
        max = 0
        min = 9999

    data['Min Words'] = c3
    data['Max Words'] = c4

    data['Min Length'] = 0
    data['Max Length'] = 0

    max = 0
    min = 9999
    c5 = []
    c6 = []
    for column in Datadf.columns:
        for i in range(len(Datadf)):
            a = str(Datadf[column][i])
            t1 = len(a)
            if(t1 > max):
                max = t1
            if(t1 < min):
                min = t1
        c5.append(min)
        c6.append(max)
        max = 0
        min = 9999

    data['Min Length'] = c5
    data['Max Length'] = c6

    data['Total Character count'] = 0
    data['Average Character count'] = 0

    c7 = []
    charcount = 0
    for column in Datadf.columns:
        for i in range(len(Datadf)):
            a = str(Datadf[column][i])
            charcount = charcount + len(a)
        c7.append(charcount)
        charcount = 0

    data['Total Character count'] = c7
    data['Average Character count'] = (
        data['Total Character count']/Datadf.shape[0])

    data['Trailing White Space'] = 0

    c8 = []
    spacecount = 0
    for column in Datadf.columns:
        for i in range(len(Datadf)):
            a = str(Datadf[column][i])
            if(a.endswith(' ')):
                spacecount = spacecount + 1
        c8.append(spacecount)
        spacecount = 0

    data['Trailing White Space'] = c8

    data['Entire White Space'] = 0
    data['Percentage Entire White Space'] = 0

    c9 = []
    spcount = 0
    for column in Datadf.columns:
        for i in range(len(Datadf)):
            if(str(Datadf[column][i]).isspace()):
                spcount = spcount+1
        c9.append(spcount)
        spcount = 0

    data['Entire White Space'] = c9
    data['Percentage Entire White Space'] = (
        data['Entire White Space']/Datadf.shape[0]) * 100

    data['Percentage Numeric'] = 0

    numcount = 0
    k = []
    for column in Datadf.columns:
        for i in range(len(Datadf)):
            if(isinstance(Datadf[column][i], numbers.Number)):
                numcount = numcount + 1
        k.append((numcount/Datadf.shape[0]) * 100)
        numcount = 0
    data['Percentage Numeric'] = k

    data['Percentage AlphaNumeric'] = 0

    charcount = 0
    k1 = []
    for column in Datadf.columns:
        for i in range(len(Datadf)):
            a = str(Datadf[column][i])
            if(a.isalnum()):
                charcount = charcount + 1
        k1.append((charcount/Datadf.shape[0]) * 100)
        charcount = 0
    data['Percentage AlphaNumeric'] = k1

    data['SpecialCharacters'] = 0

    n = 0
    c10 = []
    special_chars = ['$', '#', ',', '+', '*', '&', '^',
                     '%', '_', '-', '=', '@', '!', '(', ')', '~']
    for column in Datadf.columns:
        for i in range(len(Datadf)):

            st = str(Datadf[column][i])
            n = n + sum(v for k, v in ct.Counter(st).items()
                        if k in special_chars)
        c10.append(n)
        n = 0

    data['SpecialCharacters'] = c10
    order = ['Attributes',
             'DataTypes',
             'Count',
             'Unique',
             'Duplicates',
             'Frequency',
             'IsNull',
             'IsNotNull',
             'PercentageNull',
             'Entire_Upper',
             'Entire_Lower',
             'WithSpaces',
             'WithoutSpaces',
             'Min Words',
             'Max Words',
             'Min Length',
             'Max Length',
             'Total Character count',
             'Average Character count',
             'Trailing White Space',
             'Entire White Space',
             'Percentage Entire White Space',
             'Percentage Numeric',
             'Percentage AlphaNumeric',
             'SpecialCharacters']

    data = data[order]
    data.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\Analysis.html')  
    return data

def PatternRecog(Datadf, data): #pattern analysis function, dataset and profiling results as parameter
  pattern = []
  def change_char(s, p, r):
    return s[:p]+r+s[p+1:]
  
  for i in range(len(data)):
    if(data['Min Length'][i] - data['Max Length'][i] <= 9):
      pattern.append(data['Attributes'][i])
  
  Pattern = pd.DataFrame()
  Pattern = Datadf
  Pattern['column'] = ''
  Pattern['pattern'] = ''
  for patterncols in pattern:
    for j in range(len(Datadf)):
      a = str(Datadf.loc[Datadf.index[j] ,patterncols])
      for k in range(len(a)):
        if(a[k].isnumeric()):
          a = change_char(a, k, '9')
        if(a[k].isalpha()):
          a = change_char(a, k, 'A')
                    
      Pattern.loc[Pattern.index[j], 'column'] = Pattern.loc[Pattern.index[j], 'column'] + ' , ' + patterncols
      Pattern.loc[Pattern.index[j], 'pattern'] = Pattern.loc[Pattern.index[j], 'pattern'] + ' , ' + a
  
  for j in range(len(Datadf)):
    k = str(Pattern.loc[Pattern.index[j], 'column'])
    l = str(Pattern.loc[Pattern.index[j], 'pattern'])    
    Pattern.loc[Pattern.index[j], 'column'] = k[2:]
    Pattern.loc[Pattern.index[j], 'pattern'] = l[2:]
    
    new = Pattern['pattern'].str.split(' , ', n = 15, expand = True) 
    
  return new, pattern
  
#phone number pattern
def operations_phone(data, Pattern):
  #phone number pattern
  phone_no = []
  phone = []
  not_phone_no = []
  th = []
  mail = []
  for i in Pattern.columns:
    for j in range(len(Pattern[i])):
      if fuzz.ratio('999-999-9999', Pattern[i][j]) == 100 or fuzz.ratio('99999-99999', Pattern[i][j]) == 100 or fuzz.ratio('9999999999', Pattern[i][j]) == 100 or fuzz.ratio('99999-999999', Pattern[i][j]) == 100:
        #saving identified mail ids
        phone.append(data[i][j])
        phone_no.append([data[Pattern.columns[0]][j],data[i][j]])
        if i not in mail:
          mail.append(i)
  #daf = DataFrame(phone_no, columns = ['cust_id', 'phone_no'])
  #daf.to_csv('phone.csv')
  for i in mail:
    for j in range(len(data[i])):
      if data[i][j] not in phone:
        not_phone_no.append([data[Pattern.columns[0]][j],data[i][j]])
        
  df2 = DataFrame (phone_no,columns=['cust_id','Correct_Phone_no'])
  df2.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\ph1.html')
  df3 = DataFrame (not_phone_no,columns=['cust_id','Anomoly_Phone_no'])
  df3.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\phanomoly1.html')
  
  return phone_no, not_phone_no

  

#mail pattern Checking
def operations_mail(data, Pattern):
  #mail pattern
  mail_id = []
  not_mail_id = []
  mail = []
  for i in range(len(Pattern.columns)):
    p = 0
    for j in range(len(Pattern[Pattern.columns[i]])):
      t = fuzz.partial_ratio('A@A', Pattern[Pattern.columns[i]][j])
      if t == 100:
        try:
          if validate_email(data[Pattern.columns[i]][j]):
            p+=1
            #saving identified mail id patterns
            mail_id.append(data[Pattern.columns[i]][j])
            if Pattern.columns[i] not in mail:
              mail.append(Pattern.columns[i])
        except:
          pass
  for i in mail:
    for j in range(len(data[i])):
        if data[i][j] not in mail_id:
          not_mail_id.append(data[i][j])
          
  df10 = DataFrame (mail_id,columns=['Correct_mail'])
  df10.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\mail1.html')
  df11 = DataFrame (not_mail_id,columns=['Anomoly_mail_id'])
  df11.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\mail_anomoly1.html')
  
  return mail_id, not_mail_id

def operations_zip(data, Pattern):
  #zip code pattern
  mail = []
  zip_code = []
  not_zip_code = []
  for i in Pattern.columns:
    for j in range(len(Pattern[i])):
      if fuzz.ratio('9999.9', Pattern[i][j]) == 100 or fuzz.ratio('99999.9', Pattern[i][j]) == 100:
        #saving identified zip codes
          if i not in mail:
            mail.append(i)
          zcdb = ZipCodeDatabase()
          if len(str(data[i][j]).split('.')[0])==5 and zcdb[int(data[i][j])].zip!=None:
            zip_code.append(data[i][j])
          elif len(str(data[i][j]).split('.')[0])==4:
            nn = str(0)+str(data[i][j]).split('.')[0]
            if zcdb[nn].zip!=None:
              zip_code.append(data[i][j])
          else:
            pass
  for i in mail:
    for j in range(len(data[i])):
      if data[i][j] not in zip_code:
        not_zip_code.append(data[i][j])
        
  df6 = DataFrame (zip_code,columns=['Correct_zip_code'])
  df6.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\zip1.html')
  df7 = DataFrame (not_zip_code,columns=['Anomoly_zip_code'])
  df7.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\zipanomoly1.html')
  
  return zip_code, not_zip_code

# address pattern
def operations_address(data, Pattern):
  address = []
  not_address = []
  mail = []
  text = CommonRegex()
  for i in range(len(Pattern.columns)):
    for j in range(len(Pattern[Pattern.columns[i]])):
        t = fuzz.partial_ratio('9 A', Pattern[Pattern.columns[i]][j])
        if t == 100 and text.street_addresses(re.sub('[^A-Za-z0-9]+', ' ', data[Pattern.columns[i]][j])):
          if Pattern.columns[i] not in mail:
            mail.append(Pattern.columns[i])
          address.append(data[Pattern.columns[i]][j])
  for i in mail:
    for j in range(len(data[i])):
      if data[i][j] not in address:
          not_address.append(data[i][j])
          
  df8 = DataFrame (address,columns=['Correct_Address'])
  df8.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\address1.html')
  df9 = DataFrame (not_address,columns=['Anomoly_Address'])
  df9.to_html('P:\\priyankadata\\COEAI-LAB\\project\\Data-DQ\\djangoproject\\mysite\\templates\\addressanomoly1.html')
  
  return address, not_address
 
