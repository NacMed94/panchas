import numpy as np
import pandas as pd
import copy
import re

from IPython.display import display_html


def dictprint(d):
    print('Risky values:',*zip(d.keys(),d.values()),sep='\n\n')
    pass
    

def display_sbs(dfs_list, max_rows = 100, suffix = 'table', titles = [''],ret = False,sep = '&emsp;'):
    
    '''
    Displays dataframes as html tables and side by side (if they do not fit, they are
    represented below)
    '''
    
    # If titles list empty or not all titles given, empy titles will be assigned

    html_tables = ''
    for i, df in enumerate(dfs_list): # Iterating over all the dfs added as args
        
        if isinstance(df,pd.core.series.Series): # If df is acc series, convert to df
            df = df.to_frame()
        
        # If titles undefined or not enough titles
        if titles == [''] or i >= len(titles):
            # title equal to the string-sum of column names with commas (removing last comma with [-2]) + _suffix
            title = (df.columns.values + ', ').sum()[:-2] + ' ' + suffix
        else:
            title = titles[i]
        
        first_row = 0
        last_row = max_rows
        prints_no = int(np.ceil(len(df)/max_rows)) if len(df) >= max_rows else 1
        
        for _ in range(prints_no):
            df_s = df.iloc[first_row:last_row]
            # First converted to style, adding caption 
            df_s = df_s.style.set_table_attributes("style='display:inline'").set_caption(title)
            # Style converted to html
            html_tables += df_s.to_html()

            first_row = copy.copy(last_row)
            last_row += max_rows
            title = '<br>' # After first loop title is empty (linebreak for aligning)
        html_tables += sep # Adding separator (default "&emsp;" for tabulation) only between separate dfs
            
    # And displayed using display_html
    display_html(html_tables,raw = True)

    if ret:
        return html_tables
    else:
        pass

    
def save_as_html(html_object,filename='report.html'):
    data, metadata = get_ipython().display_formatter.format(html_object)
    with open(filename, 'w') as f:
        f.write(data['text/html']) 
    pass
    
    
def kwordcheck(df,kwordsdict,scancols = [],getbools = False):
    '''
    This code returns a df with same index as original df and with same column name 
    as the column being checked for keywords when getbools is passed as True. If False
    (default), function will return a dictionary instead based on kwordsdict and with 
    df values corresponding with passed keywords (more efficient). Kwordsdict is a 
    dictionary with the columns of dfs to assess as keys and keywords lists as values. 
    Scancols specify columns for which values will be scanned instead of exactly matched 
    (empty by default, less efficient but necessary for free-text-like variables such as 
    facility/payer/provider names). When getbools is True, the output df has boolean 
    values (True if row/column value in original df contains keyword, False otherwise). 
    This boolean df can be used to check value counts on df, remove offending rows... 
    '''
    
    # Columns of output df are the columns present in the kwordsdict
    cols = [key for key in kwordsdict.keys() if re.sub('__subcat_.+$','',key) in df.columns]
    original_cols = [re.sub('__subcat_.+$','',col) for col in cols]
    unique_values = 0
    # If getbools is True (default False)
    if getbools: 
        # Output initialised with zeroes and the index is equal to the input df index
        risky_values = pd.DataFrame(index=df.index, columns=cols)

        # Iterates over matched columns (column to check kwords on)
        for i,col in enumerate(cols):
            kwords = kwordsdict[col]
            # Original column (without the subcat) in df
            original_col = original_cols[i]
            # If column passed as one of the columns to be "scanned"
            if original_col in scancols:
                risky_values[col] = False
                # Iterates over keywords in keyword list for each value
                for word in kwords:
                    # If one of the keywords is contained in the column strings, value of risky_values is True
                    # Removing suffix from column name of original df
                    risky_values[col] = risky_values[col] | df[original_col].str.contains(word, case = False)
            else: # Looking for exact match (more efficient and default option) if no need for scanning
                # Checking against lowercase df (only if first value is string)
                if isinstance(df[original_col].dropna().iloc[0],str):
                    kwords = [word.lower() for word in kwords]
                    risky_values[col] = df[original_col].str.lower().isin(kwords)
                else:
                    risky_values[col] = df[original_col].isin(kwords)
    
    # If boolean to locate values on original df not needed (getbools=False), unique values will be returned
    else:
        # Output is a dictionary now based on kwordsdict
        keys = cols # Just for clarity
        risky_values = dict.fromkeys(keys)
        for i,ky in enumerate(keys):
            kwords = kwordsdict[ky]
            # Original column (without the subcat) in df
            original_col = original_cols[i]
            # Unique values only computed if original column is first column or different to previous one (for speed)
            unique_values = df[original_col].unique() if i==0 or original_col!=original_cols[i-1] else unique_values
            # If column passed as one of the columns to be "scanned"
            if original_col in scancols:
                risky_bool = [False]*len(unique_values)
                # Iterates over keywords in keyword list for each value
                for word in kwords:
                    # If one of the keywords is contained in the column strings, value of risky_bool is True
                    risky_bool = risky_bool | pd.Series(unique_values).str.contains(word,case=False)
                # Storing matched values on risky_values for relevant key
                risky_values[ky] = unique_values[risky_bool]
            # Looking for exact match (more efficient and default)
            else:
                # Checking against lowercase df (only if first value is string)
                if isinstance(unique_values[0],str):
                    kwords = [word.lower() for word in kwords]
                    risky_bool = np.isin(np.char.lower(unique_values.astype(str)),kwords)
                else:
                    risky_bool = np.isin(unique_values,kwords)
                risky_values[ky] = unique_values[risky_bool]

    return risky_values


def kword_dict_maker(cols,categories,values):
    ''' Standard dictionary maker for kwordcheck function '''
    d = dict()
    for col in cols:
        for i,cat in enumerate(categories):
            d[col+'__subcat_'+cat] = values[i]
    
    return d


def code_inspector(s,printrows=None,sep='\t'):
    
    unique_values = np.sort(s.dropna().unique())
    name = s.name
    print(f'Number of characters in code {name}',pd.Series(unique_values).apply(len).describe(),sep='\n',end='\n')
    print(f'\nUnique {name} codes:')
    print(*unique_values[:printrows],sep=sep,end='\n')
    pass


def get_all_codes():
  
    from reference_data.file_operations import get_all_code_categories
    
    possible_codes = set()
    for cct in get_all_code_categories():
        possible_codes.update(get_code_category_code_types(cct))
    possible_codes = list(possible_codes)
    possible_codes.sort()
    return possible_codes
    

def get_categories(code,display_dfs = True, tail = 25):
    '''
    Returns dictionary where keys are categories from the reference-data package and values are
    dataframes with codes and description of risky codes
    '''

    from reference_data.search import load_codeset
    from reference_data.file_operations import get_all_code_categories, get_code_category_code_types
    
    df_dict = dict()
    for cct in get_all_code_categories():
        if code in get_code_category_code_types(cct):
            df_dict[cct]=load_codeset(cct,code)
            df_dict[cct]=df_dict[cct].where(df_dict[cct]!='').dropna(how='all',axis=1)
    
    if display_dfs:
        display_sbs([df.tail(tail) for df in df_dict.values()],titles = [title for title in df_dict.keys()])
        
              
    return df_dict
    

def address_comparator(df,maincols,statecols,zipcols):
    
    # Dropping any row wih null values for any of the columns to make a fair comparison
    notnas_state_bool = df[[maincols[0]] + statecols].notna().all(axis=1)
    notnas_zip_bool = df[[maincols[1]] + zipcols].notna().all(axis=1)
    # Comparing state
    for col in statecols:
        print(f'{round(100*(df.loc[notnas_state_bool,maincols[0]] == df.loc[notnas_state_bool,col]).mean(),2)}% of states coincide between {maincols[0]} and {col}')
    
    # Comparing ZIP code digits (up to 3)
    for col in zipcols:
        print()
        for digits in range(3):
            print(f'{round(100*(df.loc[notnas_state_bool,maincols[1]].str[digits] == df.loc[notnas_state_bool,col].str[digits]).mean(),2)}% of {digits+1}-digit ZIPs coincide between {maincols[1]} and {col}')

    pass


def get_zip3state():

    ''' Returns df with ZIPs in one column and their apolcor state in the other. Fisher's Island only has NY'''

    from reference_data.search import load_codeset

    state_zip3 = load_codeset('state_zip3')
    # Taking all zips from row 52 of zip3_state (skipping first as empty string)
    zip3_state = pd.DataFrame(data = np.unique(state_zip3.loc[52].zips[1:]),columns = ['zips'])
    # First apply makes boolean of states for each zip. Second apply returns correct state(s)
    zip3_state['state'] = zip3_state.zips.apply(lambda zip: [zip in zips for zips in state_zip3.zips.loc[:51]]).apply(lambda boolean: state_zip3.loc[:51].state.values[boolean])
    # Unnesting and removing NY from Fisher's Island ZIP code (063)
    zip3_state['state'] = zip3_state.state.apply(lambda row: row[0])

    return zip3_state


def check_state(df,statecol = 'State'):
    
    from reference_data.search import load_codeset

    valid_states = load_codeset('state_zip3').state
    
    display_sbs([df[statecol].value_counts()],max_rows=10)
    nonapolcor_states = df[statecol].dropna().unique()[~np.isin(df[statecol].dropna().unique(),valid_states)]
    print('Non-apolcor and invalid states:', end = ' ')
    print(*nonapolcor_states,sep=', ')
    
    pass

    
def check_zips(df,zipcol = 'Zip',statecol = 'State',df_display = True):
    
    from reference_data.search import load_codeset
    
    valid_zips = load_codeset('state_zip3').zips.loc[52][1:]
    nonapolcor_zips_bool = (~df[zipcol].isin(valid_zips).values) & df[zipcol].notna() 
    
    # If no invalid/non-apolcor ZIP codes are present (all in bool are False), print so and return None
    if not nonapolcor_zips_bool.any():
        print('No non-apolcor ZIPs present')
        return None
    
    if df_display:
        
        # Dropping null ZIPs (to visualise better) and filling null states to visualise nonapolcor ZIPs on empty states
        nonapolcor_zips_s = df.loc[nonapolcor_zips_bool,[zipcol,statecol]].fillna('no_state').groupby(statecol).value_counts()
        display(pd.DataFrame(nonapolcor_zips_s.rename('ZIP_value_counts')))
        print("Non-apolcor and invalid ZIPs:", end = ' ')
        print(*nonapolcor_zips_s.index.droplevel(0).unique(),sep=', ')
    
    return nonapolcor_zips_bool


def check_zipstate_misal(df, zipcol,statecol):

    """This function checks if there are any rows with misaligned values for ZIP and state."""

    from reference_data.search import load_codeset

    zip3_state = get_zip3state()
    # Not taking into account non-apolcor (and invalid and null) ZIPs and states
    misal_bool = df[zipcol].isin(zip3_state.zips) & df[statecol].isin(zip3_state.state.unique()) & (~df[[zipcol,statecol]].astype(str).sum(axis=1).isin(zip3_state.sum(axis=1).values))

    print(f'There are {misal_bool.sum()} rows with misaligned ZIP and state')

    return misal_bool


def smallzip_merger(df, zipcol, inplace=True):
    ''' 
    Finds small ZIPs and merges them with neighbour with format SMALLZIP_NEIGHBOUR
    Uses kwordcheck twice (would be faster if integrated but quick solution).
    Inplace default is True because I like to live dangerously.
    '''
    from reference_data.search import load_codeset
    from panchas import kwordcheck
    
    zip3_small = load_codeset('zip3_small')
    # Not taking into account ZIP code 576 whose population is larger than 20K
    zip3_small = zip3_small.loc[zip3_small.pop_less_than_20000 == 'true',['code','neighbor']]

    # We find list of relevant ZIP codes and their neighbours
    smallzips_values = list(kwordcheck(df,{zipcol:zip3_small.code})[zipcol])
    # 879 is the neighbour ZIP of 878 and vice versa, so if both are in the data, we only need to check and merge for one
    smallzips_values.remove('878') if '878' in smallzips_values and '879' in smallzips_values else None

    # Making df to get dictionary for kwordcheck
    smallzips_neighbours = zip3_small[zip3_small.code.isin(smallzips_values)]

    # Tweaking df to format dictionary so it can be used in function
    smallzips_neighbours_kwords = smallzips_neighbours.set_index('patient_adr_zip__subcat_' \
        + smallzips_neighbours.code,drop=False).apply(list,axis=1).to_dict()

    # Finding bools
    smallneighzips_dfbool = kwordcheck(df,smallzips_neighbours_kwords,getbools=True)

    print(f'Editing a total of {smallneighzips_dfbool.sum().sum()} rows with small and neighbouring ZIP codes',end = "\n\n")
    
    # Making a copy if inplace is False
    df_out = df if inplace else df.copy()
        
    for i, col in enumerate(smallneighzips_dfbool.columns):
        small = smallzips_neighbours_kwords[col][0]
        neighbour = smallzips_neighbours_kwords[col][1]
        
        print(f'Merging {smallneighzips_dfbool[col].sum()} rows with small ZIP {small} or neighbouring ZIP {neighbour}')
        df_out[zipcol].where(~smallneighzips_dfbool[col], small + '_' + neighbour, inplace = True)
    
    return df_out if not inplace else None


def column_merger(df,*categories):
    
    for cat in categories:
        catcols = df.columns[df.columns.str.contains(cat,case=False)]
        df[cat] = df[catcols].any(axis=1).rename(cat)
        df.drop(columns = catcols,inplace = True)
        

def read_folder_dfs(folderpath, keys = [], filetype='csv', delimiter=',', index_col=0, dfs_display=True):
    
    ''' Reads all .csv or .pkl files in folder and stores them in global variables. '''
    
    from os import listdir
    fnames = listdir(folderpath)
    fnames.sort()
    df_dict = dict()
    for i,file in enumerate(fnames):
        
        if keys == []:
            key = re.search('.+[.]',file)[0][:-1]
        else:
            key = keys[i]
               
        # Reading df into global variable with varname as name
        filepath = folderpath + '/' + file
        if filetype == 'csv':
            df_dict[key] = pd.read_csv(filepath, delimiter=delimiter,index_col=index_col) # Setting delimiter and index column
        elif filetype == 'pkl':
            df_dict[key] = pd.read_pickle(filepath) # Setting delimiter and index column
        else:
            print("Sorry, can't read that")
            return None

        print(f'Saving file {filepath} into dictionary with key {key}')
        if dfs_display:
            display(df_dict[key].head(3))
        print('\n')
        
    return df_dict
