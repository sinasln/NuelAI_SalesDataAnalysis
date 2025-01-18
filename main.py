

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from transformers import pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import requests
import ast
import json
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import time
import matplotlib.pyplot as plt
import altair as alt
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
import folium
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import os

# Function to save dictionaries to a file in the 'Data' folder
def save_dictionaries_to_file(customer_dict,season_dict, store_dict, store_dict_sold_items, store_dict_holidays, store_dict_customer_count, payment_method_dict,transactions):
    data_folder = 'Data'  # Folder to store the pickle file

    # Create the folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # File path to save the dictionaries
    file_path = os.path.join(data_folder, 'dictionaries.pkl')

    # Save the dictionaries as a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump({
            'customer_dict':customer_dict,
            'season_dict': season_dict,
            'store_dict': store_dict,
            'store_dict_sold_items': store_dict_sold_items,
            'store_dict_holidays': store_dict_holidays,
            'store_dict_customer_count': store_dict_customer_count,
            'payment_method_dict': payment_method_dict,
            'transactions':transactions
        }, f)

# Function to load dictionaries from the pickle file
def load_dictionaries_from_file():
    data_folder = 'Data'  # Folder to store the pickle file

    # File path to read the dictionaries from
    file_path = os.path.join(data_folder, 'dictionaries.pkl')

    # Check if the file exists and load the dictionaries
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None  # Return None if the file doesn't exist


def save_dictionaries_to_session(season_dict, store_dict, store_dict_sold_items, store_dict_holidays, store_dict_customer_count, payment_method_dict):
    st.session_state['season_dict'] = season_dict
    st.session_state['store_dict'] = store_dict
    st.session_state['store_dict_sold_items'] = store_dict_sold_items
    st.session_state['store_dict_holidays'] = store_dict_holidays
    st.session_state['store_dict_customer_count'] = store_dict_customer_count
    st.session_state['payment_method_dict'] = payment_method_dict

def load_dictionaries_from_session():
    if 'season_dict' in st.session_state:
        return st.session_state['season_dict'], st.session_state['store_dict'], st.session_state['store_dict_sold_items'], st.session_state['store_dict_holidays'], st.session_state['store_dict_customer_count'], st.session_state['payment_method_dict']
    return None, None, None, None, None, None


def update_progress(current_step: int, total_steps: int,progress_bar,status_text):
    progress = current_step / total_steps
    progress_bar.progress(progress)
    status_text.text(f"Progress: {progress*100:.2f} % completed.")

            
def avg_spending(df,column_name1,column_name2):
    unique_count = df[column_name1].nunique()
    column_sum = df[column_name2].sum()
    return column_sum/unique_count

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config
    except Exception as e:
        st.error(f"Error reading JSON file: {e}")
        return None
def check_proper_case(name):
    if pd.isnull(name):  # Handle missing values
        return False
    return all(word.istitle() for word in name.split())

def read_data(source_path):
    
    data = pd.read_csv(source_path)
    print("Data read successfully")
    return data
def handling_null(data):
    condition = pd.isna(data['Promotion'])
    data.loc[condition, "Promotion"] = "NA"
    
    return data 

def handle_missing_values(df, action, column_defaults=None):
    if action == 'Drop rows with missing values':
        print("entered drop")
        return df.dropna()
    elif action == 'Replace missing values with default':
        for column, default_value in column_defaults.items():
            
            correct_type = df[column].dtype
            try:
                if pd.api.types.is_numeric_dtype(correct_type):
                    default_value = pd.to_numeric(default_value, errors='coerce')
                elif pd.api.types.is_datetime64_any_dtype(correct_type):
                    default_value = pd.to_datetime(default_value, errors='coerce')
                elif pd.api.types.is_bool_dtype(correct_type):
                    default_value = default_value.lower() in ['true', '1', 'yes']
                else:  
                    default_value = str(default_value)
            except Exception as e:
                st.error(f"Invalid default value '{default_value}' for column '{column}': {e}")
                continue
            
            df[column] = df[column].fillna(default_value)
        return df
    else:
        return df

def handle_duplicates(df, remove_duplicates=False):
    if remove_duplicates:
        # Keep only the first occurrence of each duplicated row
        df = df.drop_duplicates(keep='first')
    return df
def adding_holidays(data,json_data):

    progress_container = st.container()
    progress_bar = progress_container.progress(0)  # Create progress bar
    status_text = progress_container.empty()  # Create space for status message
        
    api_url = json_data.get("api_url")
    params = json_data.get("params")
    min_date = data["Date"].min()
    max_date = data["Date"].max()
    
    cntr = 0
    for year in range(min_date.year,max_date.year+1):
        
        update_progress(cntr,max_date.year+1 - min_date.year,progress_bar,status_text)
        cntr+=1
        params['year'] = year
        response = requests.get(api_url, params=params)
    
        # Check if request was successful
        if response.status_code == 200:
            r_data = response.json()  # Parse JSON data
            holidays = r_data["response"]["holidays"]  # Extract holidays list
            
            # Create a DataFrame
            holidays_df = pd.DataFrame(holidays)
            
            # Display columns in the DataFrame
            
            
            # Select and display relevant information
            if not holidays_df.empty:
                for indx,row in enumerate(holidays_df['date']):
                    
                        
                    y = row['datetime']['year']
                    m = row['datetime']['month']
                    d = row['datetime']['day']
                    condition = (data["Date"].dt.year == y) & (data["Date"].dt.month == m) & (data["Date"].dt.day == d)
        
                    # Modify a column value where the condition is True
                    data.loc[condition, "Holiday"] = True  # Update "Value" to 999
            else:
                print("No holidays found!")
        else:
            print(f"Error fetching data: {response.status_code}")
    update_progress(max_date.year+1 - min_date.year,max_date.year+1 - min_date.year,progress_bar,status_text)
    progress_container.empty()
    return data
        
        
def EDA(data):


    unique_values = data['Season'].unique()
    season_dict = dict()
    for itm in unique_values:
        season_dict[itm] = 0

    unique_values = data['Customer_Name'].unique()
    customer_dict = dict()
    for itm in unique_values:
        customer_dict[itm] = {'Spend':0 , 'Shopping_Count':0,'Job':''}

    unique_values = data['Store_Type'].unique()
    store_dict = dict()
    for itm in unique_values:
        store_dict[itm] = {'Winter':0,'Spring':0,'Fall':0,'Summer':0}

    unique_values = data['Store_Type'].unique()
    store_dict_sold_items = dict()
    for itm in unique_values:
        store_dict_sold_items[itm] = dict()

    unique_values = data['Store_Type'].unique()
    store_dict_holidays = dict()
    for itm in unique_values:
        store_dict_holidays[itm] = {'normal':0 , 'holiday':0 , 'cnt_normal':0 , 'cnt_holiday':0}

    unique_values = data['Store_Type'].unique()
    store_dict_customer_count = dict()
    for itm in unique_values:
        store_dict_customer_count[itm] = 0
    unique_values2 = data['Payment_Method'].unique()
    payment_method_dict = dict()
    for itm in unique_values:
        payment_method_dict[itm] = {}
        for itm2 in unique_values2:
            payment_method_dict[itm][itm2] = 0
    transactions = []
    progress_container = st.container()
    progress_bar = progress_container.progress(0)  # Create progress bar
    status_text = progress_container.empty()  # Create space for status message
    for i in range (0,len(data)):
        if i%1000 == 0:
            update_progress(i,len(data),progress_bar,status_text)
        tmp_list = ast.literal_eval(data['Product'][i])
        transactions.append(tmp_list)
        season_dict[data['Season'][i]]+=1
        payment_method_dict[data['Store_Type'][i]][data['Payment_Method'][i]]+=1
        store_dict[data['Store_Type'][i]][data['Season'][i]]+=1
        customer_dict[data['Customer_Name'][i]]['Spend']+=data['Total_Cost'][i]
        customer_dict[data['Customer_Name'][i]]['Shopping_Count']+=1
        customer_dict[data['Customer_Name'][i]]['Job']=data['Customer_Category'][i]
        for itm in tmp_list:
            if itm in store_dict_sold_items[data['Store_Type'][i]]:
                store_dict_sold_items[data['Store_Type'][i]][itm]+=1
            else:
                store_dict_sold_items[data['Store_Type'][i]][itm]=1
        store_dict_customer_count[data['Store_Type'][i]]+=1
        if data['Holiday'][i]:
            store_dict_holidays[data['Store_Type'][i]]['holiday']+=data['Total_Cost'][i]
            store_dict_holidays[data['Store_Type'][i]]['cnt_holiday']+=1
        else:
            store_dict_holidays[data['Store_Type'][i]]['normal']+=data['Total_Cost'][i]
            store_dict_holidays[data['Store_Type'][i]]['cnt_normal']+=1
    update_progress(len(data),len(data),progress_bar,status_text)
    progress_container.empty()
    return customer_dict,season_dict,store_dict,store_dict_sold_items,store_dict_holidays,store_dict_customer_count,payment_method_dict,transactions  

def is_subset(itemset, transaction):
    return set(itemset).issubset(transaction)
def compare_itemset(list1,transaction_list):
    count1 = 0
    
    for transaction in transaction_list:
        if is_subset(list1, transaction):
            count1+=1
        
    return count1
if 'computing_done' not in st.session_state:
    st.session_state.computing_done= 0

def main():
    
    if st.session_state.computing_done==0:
        st.session_state.computing_done=1
        st.title("Test App")
        st.subheader("Reading main data file(s)")
        data = read_data("./Data/Retail_Transactions_Dataset2.csv")
        
        if data is not None:
        
            st.write("Data Preview (First 5 rows):")
            st.dataframe(data.head()) 
            data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d %H:%M:%S")
            data["Holiday"] = False
            st.subheader("Integrating Holiday Data")
            with open('./Data/parameters.json', 'r') as file:
                json_data = json.load(file)
            
            data = adding_holidays(data,json_data)
            st.dataframe(data)
            data.to_csv('./Data/data.csv', index=False)
            pp_button = st.button("Go to the pre-processing section")

            if pp_button:
                st.session_state.computing_done=1
    elif st.session_state.computing_done==1:
        data = read_data("./Data/data.csv")
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d %H:%M:%S")
        
        data = handling_null(data)
        
        
        
        st.title("DataFrame Filter App")
        st.write("Cleaning the data and missing values:")
        
        
        st.subheader("Missing Values")
        missing_values = data.isnull().sum()
        st.write(missing_values)

        
        action = st.selectbox("Choose action for missing values", 
                                ['Drop rows with missing values', 'Replace missing values with default'])
        
        if action == 'Replace missing values with default':
            column_defaults = {}
            for column in data.columns:
                if missing_values[column] > 0: 
                    default_value = st.text_input(f"Enter default value for '{column}'")
                    
                    
                    column_defaults[column] = default_value  
            
            data = handle_missing_values(data, action, column_defaults)
        else:
            data = handle_missing_values(data, action)


        
        st.subheader("Processed Data")
        st.dataframe(data)
        data['Discount_Applied'] = data['Discount_Applied'].map(lambda x: str(x).strip().lower() == 'true' if pd.notna(x) else None)
        data['Discount_Applied'] = data['Discount_Applied'].astype('boolean')  
       
        for indx,column in enumerate(data.columns):
            if data.dtypes[indx] == object:
                data[column] = data[column].str.title()
        
      
    
        st.subheader("Duplicated Rows")
        duplicates = data[data.duplicated()]
        st.write(f"Number of duplicated rows: {len(duplicates)}")
        st.write(f"len before : {len(data)}")

       
        remove_duplicates = st.checkbox("Remove extra duplicate rows and keep only the first occurrence")
        if remove_duplicates:
            data = handle_duplicates(data, remove_duplicates=True)
            st.subheader("Data After Removing Extra Duplicates")
            st.dataframe(data)
        st.write(f"len after : {len(data)}")

        
        st.sidebar.header("Filter Options")

       
        st.sidebar.subheader("Numeric Filters")
        transaction_id_min, transaction_id_max = st.sidebar.slider(
            "Transaction_ID Range",
            int(data['Transaction_ID'].min()), int(data['Transaction_ID'].max()),
            (int(data['Transaction_ID'].min()), int(data['Transaction_ID'].max()))
        )

        total_items_min, total_items_max = st.sidebar.slider(
            "Total_Items Range",
            int(data['Total_Items'].min()), int(data['Total_Items'].max()),
            (int(data['Total_Items'].min()), int(data['Total_Items'].max()))
        )

        total_cost_min, total_cost_max = st.sidebar.slider(
            "Total_Cost Range",
            float(data['Total_Cost'].min()), float(data['Total_Cost'].max()),
            (float(data['Total_Cost'].min()), float(data['Total_Cost'].max()))
        )

       
        st.sidebar.subheader("Date Filter")
        date_min, date_max = st.sidebar.date_input(
            "Date Range",
            (data['Date'].min(), data['Date'].max())
        )

        
        st.sidebar.subheader("Categorical Filters")
        holiday_filter = st.sidebar.multiselect("Holiday", options=data['Holiday'].unique(), default=data['Holiday'].unique())
        promotion_filter = st.sidebar.multiselect("Promotion", options=data['Promotion'].unique(), default=data['Promotion'].unique())
        season_filter = st.sidebar.multiselect("Season", options=data['Season'].unique(), default=data['Season'].unique())
        customer_category_filter = st.sidebar.multiselect("Customer_Category", options=data['Customer_Category'].unique(), default=data['Customer_Category'].unique())
        city_filter = st.sidebar.multiselect("City", options=data['City'].unique(), default=data['City'].unique())
        payment_method_filter = st.sidebar.multiselect("Payment_Method", options=data['Payment_Method'].unique(), default=data['Payment_Method'].unique())

       
        filtered_df = data[
            (data['Transaction_ID'] >= transaction_id_min) & (data['Transaction_ID'] <= transaction_id_max) &
            (data['Total_Items'] >= total_items_min) & (data['Total_Items'] <= total_items_max) &
            (data['Total_Cost'] >= total_cost_min) & (data['Total_Cost'] <= total_cost_max) &
            (data['Date'] >= pd.Timestamp(date_min)) & (data['Date'] <= pd.Timestamp(date_max)) &
            (data['Holiday'].isin(holiday_filter)) &
            (data['Promotion'].isin(promotion_filter)) &
            (data['Season'].isin(season_filter)) &
            (data['Customer_Category'].isin(customer_category_filter)) &
            (data['City'].isin(city_filter)) &
            (data['Payment_Method'].isin(payment_method_filter))
        ]

       
        st.write("Filtered DataFrame")
        st.dataframe(filtered_df)


     
        
        
        compute_button = st.button("Approve processed data")

        if compute_button:
            filtered_df.to_csv('./Data/data.csv', index=False)
            
            st.session_state.computing_done = 2
            visualize_button = st.button("Go to EDA Section")

            if visualize_button:
                st.write(f"data Visualization")
                        

                
                
    elif  st.session_state.computing_done==2:
        data = read_data("./Data/data.csv")
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d %H:%M:%S")
        st.subheader("Exploratory Data Analysis (EDA)")
        
        
        st.session_state.computing_done = 3
       
        customer_dict,season_dict, store_dict, store_dict_sold_items, store_dict_holidays, store_dict_customer_count, payment_method_dict,transactions = EDA(data)
        save_dictionaries_to_file(customer_dict,season_dict, store_dict, store_dict_sold_items, store_dict_holidays, store_dict_customer_count, payment_method_dict,transactions)
        city_costs = data.groupby('City')['Total_Cost'].sum().reset_index()

        geolocator = Nominatim(user_agent="geoapi")
        latitudes = []
        longitudes = []

        for city in city_costs['City']:
            location = geolocator.geocode(f"{city}, USA")
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
            else:
                latitudes.append(None)
                longitudes.append(None)
            time.sleep(1)  

        city_costs['latitude'] = latitudes
        city_costs['longitude'] = longitudes

       
        city_costs = city_costs.dropna(subset=['latitude', 'longitude'])

        
        m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)  
        heat_data = city_costs[['latitude', 'longitude', 'Total_Cost']].values.tolist()
        HeatMap(heat_data).add_to(m)

        
        m.save("./Data/us_city_heatmap.html")
        compute_button = st.button("Apply EDA and generate the heatmap")

        if compute_button:
            
            st.session_state.computing_done = 3
            visualize_button = st.button("Visualize data")

            if visualize_button:
                st.write(f"data Visualization")

    elif  st.session_state.computing_done==3:
        data = read_data("./Data/data.csv")
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d %H:%M:%S")

        

        
        
        dictionaries = load_dictionaries_from_file()

        
        
        customer_dict = dictionaries['customer_dict']
        season_dict2 = dictionaries['season_dict']
        store_dict2 = dictionaries['store_dict']
        store_dict_sold_items2 = dictionaries['store_dict_sold_items']
        store_dict_holidays = dictionaries['store_dict_holidays']
        store_dict_customer_count = dictionaries['store_dict_customer_count']
        payment_method_dict = dictionaries['payment_method_dict']
        transactions = dictionaries['transactions']

          

        #season_dict-------------------------------------------------------------------------------------
        season_df = pd.DataFrame({
        'Season': list(season_dict2.keys()),
        'Count': list(season_dict2.values())
        })

       
        season_df['Count'] = pd.to_numeric(season_df['Count'], errors='coerce')

        
        st.title("Seasonal Sales Distribution")

        
        min_y = season_df['Count'].min()
        max_y = season_df['Count'].max()
        buffer = (max_y - min_y) * 0.05  
        y_min = min_y - buffer
        y_max = max_y + buffer

       
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(season_df['Season'], season_df['Count'], color='skyblue')
        ax.set_title("Seasonal Sales by Season", fontsize=16)
        ax.set_xlabel("Season", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_ylim([y_min, y_max])
        ax.grid(axis='y', linestyle='--', alpha=0.7)

       
        st.pyplot(fig)


        #store_dict-------------------------------------------------------------------------------------

        store_type_options = list(store_dict2.keys())
        st.title("Store Seasonal Sales Distribution")

       
        if 'selected_store' not in st.session_state:
            st.session_state.selected_store = store_type_options[0] 

        
        selected_store = st.selectbox("Select Store Type", store_type_options, index=store_type_options.index(st.session_state.selected_store), key="selectbox1")

    
        if selected_store != st.session_state.selected_store:
            st.session_state.selected_store = selected_store

        
        store_sales = store_dict2[st.session_state.selected_store]

        
        season_df_for_store = pd.DataFrame(list(store_sales.items()), columns=["Season", "Sales"])

       
        fig, ax = plt.subplots()
        ax.bar(season_df_for_store['Season'], season_df_for_store['Sales'], color='skyblue')

       
        min_y = min(season_df_for_store['Sales'])
        max_y = max(season_df_for_store['Sales'])

       
        buffer = (max_y - min_y) * 0.05 
        ax.set_ylim(min_y - buffer, max_y + buffer)

       
        ax.set_xlabel('Season')
        ax.set_ylabel('Sales Count')
        ax.set_title(f'Sales per Season for {st.session_state.selected_store}')
        ax.set_xticklabels(season_df_for_store['Season'], rotation=45)

      
        st.pyplot(fig)
        #------------------------------------------------------------------------
        store_type_options = list(store_dict_sold_items2.keys())
        selected_store = st.selectbox("Select Store Type", store_type_options, key="selectbox2")

       
        store_sales = store_dict_sold_items2[selected_store]

       
        sorted_items = sorted(store_sales.items(), key=lambda x: x[1], reverse=True)[:10]

     
        top_items_df = pd.DataFrame(sorted_items, columns=["Item", "Sold Count"])

      
        st.title(f"Top 10 Sold Items in {selected_store}")
        st.bar_chart(top_items_df.set_index('Item')['Sold Count'])

     
        st.write(top_items_df)
        #----------------------------------------------------------------------
        store_types = list(store_dict_holidays.keys())
        holiday_data = []
        normal_data = []

        for store_type in store_types:
            holiday_revenue = store_dict_holidays[store_type]['holiday']
            normal_revenue = store_dict_holidays[store_type]['normal']
            
            holiday_count = store_dict_holidays[store_type]['cnt_holiday']
            normal_count = store_dict_holidays[store_type]['cnt_normal']
            
           
            avg_holiday_revenue = holiday_revenue / holiday_count if holiday_count > 0 else 0
            avg_normal_revenue = normal_revenue / normal_count if normal_count > 0 else 0
            
            holiday_data.append(avg_holiday_revenue)
            normal_data.append(avg_normal_revenue)

       
        fig, ax = plt.subplots(figsize=(10, 6))

      
        bar_width = 0.35
        x_positions = np.arange(len(store_types))

       
        ax.bar(x_positions - bar_width / 2, holiday_data, bar_width, label='Holiday Sales', color='green')
        ax.bar(x_positions + bar_width / 2, normal_data, bar_width, label='Normal Sales', color='blue')

      
        ax.set_xlabel('Store Type')
        ax.set_ylabel('Average Revenue')
        ax.set_title('Average Revenue for Holiday and Normal Days by Store Type')

       
        ax.set_xticks(x_positions)
        ax.set_xticklabels(store_types, rotation=45, ha='right')

        
        ax.legend()

       
        min_y = min(min(holiday_data), min(normal_data))
        max_y = max(max(holiday_data), max(normal_data))
        buffer = (max_y - min_y) * 0.05  

        ax.set_ylim(min_y - buffer, max_y + buffer)

        
        plt.tight_layout()
        st.pyplot(fig)
        #-----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))

       
        ax.bar(store_dict_customer_count.keys(), store_dict_customer_count.values(), color='skyblue')

       
        ax.set_xlabel('Store Type')
        ax.set_ylabel('Customer Count')
        ax.set_title('Customer Count for Each Store Type')

       
        ax.set_xticklabels(store_dict_customer_count.keys(), rotation=45, ha="right")

       
        y_min = min(store_dict_customer_count.values())
        y_max = max(store_dict_customer_count.values())

       
        buffer = (y_max - y_min) * 0.05  
        ax.set_ylim(y_min - buffer, y_max)

       
        plt.tight_layout()
        st.pyplot(fig)
        #---------------------------------------------------------------------
        store_type_options = list(payment_method_dict.keys())
        selected_store_type = st.selectbox("Select Store Type", store_type_options)

        
        store_payment_methods = payment_method_dict[selected_store_type]

       
        payment_methods = list(store_payment_methods.keys())
        payment_counts = list(store_payment_methods.values())

      
        fig, ax = plt.subplots(figsize=(6, 6))  

      
        ax.pie(payment_counts, labels=payment_methods, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'skyblue', 'gold'])
        ax.axis('equal')  
        ax.set_title(f"Payment Method Distribution for {selected_store_type}")

        
        st.pyplot(fig)

        st.title("Product Co-Purchase Finder")

       
        st.subheader("Select Products to Check Co-Purchase Occurrence")
        unique_products = set(product for sublist in transactions 
                            for subsublist in (sublist if isinstance(sublist[0], list) else [sublist]) 
                            for product in subsublist)

        selected_products = st.multiselect("Choose products:", options=sorted(unique_products))

        if st.button("Find Co-Purchase Occurrence"):
            if selected_products:
                occurrence_count = compare_itemset(selected_products, transactions)
                st.write(f"The itemset {selected_products} was purchased together {occurrence_count} times.")
                
               
                if 'results' not in st.session_state:
                    st.session_state['results'] = []
                st.session_state['results'].append({'Itemset': selected_products, 'Occurrence': occurrence_count})

       
        if 'results' in st.session_state and st.session_state['results']:
            st.subheader("Saved Results")
            saved_results_df = pd.DataFrame(st.session_state['results'])

          
            for i, row in saved_results_df.iterrows():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"Itemset: {row['Itemset']}, Occurrence: {row['Occurrence']}")
                with col2:
                    if st.button("Delete", key=f"delete_{i}"):
                     
                        st.session_state['results'].pop(i)
        #---------------------------------------
        df = pd.DataFrame(customer_dict).T.reset_index()
        df.rename(columns={'index': 'Customer'}, inplace=True)

        st.title("Customer Spending and Shopping Analysis")

      
        average_spending = df['Spend'].mean()
        st.subheader(f"Average Spending: {average_spending:.2f}")

      
        top_buyers = df[['Customer', 'Spend']].sort_values(by='Spend', ascending=False).head(10)
        st.subheader("Top 10 Buyers by Total Spending")
        st.write(top_buyers)

       
        top_frequent_buyers = df[['Customer', 'Shopping_Count']].sort_values(by='Shopping_Count', ascending=False).head(10)
        st.subheader("Top 10 Frequent Buyers")
        st.write(top_frequent_buyers)
                        


        st.title("Customer Spending and Job-Type Analysis")

        



        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)


        job_options = df_shuffled['Job'].unique().tolist()
        selected_jobs = st.multiselect("Select Job Types to Compare", job_options, default=job_options[:2])  # Default: first 2 jobs

      
        filtered_df = df_shuffled[df_shuffled['Job'].isin(selected_jobs)]

        if filtered_df.empty:
            st.warning("No data available for the selected job types. Please select at least one job type.")
        else:
           
            job_palette = sns.color_palette("Set2", n_colors=len(filtered_df['Job'].unique()))
            job_colors = dict(zip(filtered_df['Job'].unique(), job_palette))

           
            st.subheader("Customer Spending Plot (Selected Job Types)")
            fig, ax = plt.subplots(figsize=(15, 8))

          
            for job in filtered_df['Job'].unique():
                job_group = filtered_df[filtered_df['Job'] == job]
                ax.scatter(job_group.index, job_group['Spend'], label=job, color=job_colors[job], s=40, alpha=0.7)

           
            ax.set_title("Customers' Spending by Selected Job Types")
            ax.set_xlabel("Customer Index")
            ax.set_ylabel("Spending")
            ax.legend(title="Job Type", bbox_to_anchor=(1.05, 1), loc='upper left')

            
            ax.set_xticks(filtered_df.index)
            ax.set_xticklabels(filtered_df.index)

            
            ax.set_xticks([])

          
            st.pyplot(fig)


        st.title("Transaction Total Cost by Payment Method")

        




        payment_method_options = data['Payment_Method'].unique().tolist()
        default_methods = ['Cash', 'Credit Card']
        selected_methods = st.multiselect(
            "Select Payment Methods to Display",
            payment_method_options,
            default=[method for method in default_methods if method in payment_method_options]  
        )

        
        filtered_data = data[data['Payment_Method'].isin(selected_methods)]

        if filtered_data.empty:
            st.warning("No data available for the selected payment methods. Please select at least one payment method.")
        else:
            
            payment_method_palette = sns.color_palette("Set2", n_colors=len(filtered_data['Payment_Method'].unique()))
            payment_method_colors = dict(zip(filtered_data['Payment_Method'].unique(), payment_method_palette))

            
            st.subheader("Transactions Plot by Payment Method")
            fig, ax = plt.subplots(figsize=(15, 8))

            
            for payment_method in filtered_data['Payment_Method'].unique():
                payment_group = filtered_data[filtered_data['Payment_Method'] == payment_method]
                ax.scatter(payment_group['Transaction_ID'], payment_group['Total_Cost'],
                        label=payment_method, color=payment_method_colors[payment_method], s=40, alpha=0.7)

           
            ax.set_title("Transactions and Total Cost by Payment Method")
            ax.set_xlabel("Transaction Number")
            ax.set_ylabel("Total Cost")
            ax.legend(title="Payment Method", bbox_to_anchor=(1.05, 1), loc='upper left')

           
            plt.xticks([])
            st.pyplot(fig)
        compute_button = st.button("Approve")

        if compute_button:
            
            st.session_state.computing_done = 4
            visualize_button = st.button("Go to the Advanced Data Analysis")

            if visualize_button:
                st.session_state.computing_done = 4
    elif st.session_state.computing_done==4 : 
        data = read_data("./Data/data.csv")
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d %H:%M:%S")
        data['Total_Cost'] = pd.to_numeric(data['Total_Cost'], errors='coerce')

       
        store_revenue = data.groupby('Store_Type')['Total_Cost'].sum().reset_index()

     
        sorted_store_revenue = store_revenue.sort_values(by='Total_Cost', ascending=False)

       
        best_stores = sorted_store_revenue.head(3)
        worst_stores = sorted_store_revenue.tail(3)

       
        best_data = data[data['Store_Type'].isin(best_stores['Store_Type'])]
        worst_data = data[data['Store_Type'].isin(worst_stores['Store_Type'])]

       
        best_store_summary = best_data.describe(include='all')
        worst_store_summary = worst_data.describe(include='all')

        
        best_store_summary, worst_store_summary
        store_types = ['All'] + data['Store_Type'].unique().tolist()
        selected_store = st.selectbox("Select Store Type", store_types,key = "selctionBar_store")

        
        if selected_store != 'All':
            filtered_data = data[data['Store_Type'] == selected_store]
        else:
            filtered_data = data

       
        aggregated_data = filtered_data.groupby('Date')['Total_Cost'].sum().reset_index()

        
        st.write(f"### Total Cost Over Time ({selected_store})")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(aggregated_data['Date'], aggregated_data['Total_Cost'], marker='o', linestyle='-')
        ax.set_title(f"Total Cost Trend ({selected_store})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Cost")
        ax.grid(True)
        st.pyplot(fig)
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        st.title("Correlation Matrix")

        
        

        numeric_columns = data.select_dtypes(include=['number']).columns.difference(['Transaction_ID'])
        
        correlation_matrix = data[numeric_columns].corr()  

        
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        #-----------------------------------------------------------
        st.title("Regression Analysis - Store Sales")


        

       
        data = pd.get_dummies(data, columns=['Payment_Method', 'City', 'Store_Type', 'Customer_Category', 'Season', 'Promotion', 'Holiday'], drop_first=True)
        
       
        X = data[['Total_Items']+['Year']+['Month']+['Day'] + [col for col in data if col.startswith('Payment_Method')]+ [col for col in data if col.startswith('City')]+ [col for col in data if col.startswith('Store_Type')]
        + [col for col in data if col.startswith('Customer_Category')]+ [col for col in data if col.startswith('Season')]+ [col for col in data if col.startswith('Promotion')]+ [col for col in data if col.startswith('Holiday')]] 
        y = data['Total_Cost']
        
        X = X.astype(int)
        
       
        X = sm.add_constant(X)

        
        model = sm.OLS(y, X).fit()

        
        st.write("### Regression Model Summary")
        st.write(model.summary())

       
        residuals = model.resid
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color='blue', ax=ax)
        st.write("### Residuals Distribution")
        st.pyplot(fig)
        #---------------------------------------------
        

       
        vif_data = pd.DataFrame()
        vif_data['Variable'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

        st.write("### Variance Inflation Factor (VIF)")
        st.write(vif_data)
        #_________________________________________
        




        model_name = "facebook/bart-large-cnn" 
        summarizer = pipeline("summarization", model=model_name)
        

        
        correlation_dict = correlation_matrix.to_dict()
        vif_dict = vif_data.set_index('Variable').to_dict()['VIF']
        ols_summary = model.summary().as_text()

        
        analysis_input = f"""
        ### Correlation Matrix:
        {correlation_dict}

        ### VIF Values:
        {vif_dict}

        ### OLS Summary:
        {ols_summary}

        Please provide a detailed analysis of the above results. Point out any issues with multicollinearity, significant correlations, and any important findings from the regression analysis.
        """
        
        max_input_length = 1024  
        if len(analysis_input) > max_input_length:
            analysis_input = analysis_input[:max_input_length]
        
        try:
            summary = summarizer(analysis_input, max_length=512, min_length=50, do_sample=False)[0]['summary_text']
            st.write("### AI Analysis Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error during analysis: {e}")












       
        data_2 = {
            'correlation_matrix': correlation_dict,
            'vif': vif_dict,
            'ols_summary': ols_summary
        }

        
        api_url = "http://127.0.0.1:8000/analyze/"
        response = requests.post(api_url, json=data_2)

       
        if response.status_code == 200:
            analysis_result = response.json()  
            st.write("### AI-Generated Insights")
            st.write(analysis_result)
        else:
            st.write("Error in sending data to the analysis API.")
        compute_button = st.button("Go next")
        
        if compute_button:
            
            st.session_state.computing_done = 5
            visualize_button = st.button("Go to the Advanced Data Analysis")

        

    else: 
        data = read_data("./Data/data.csv")
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d %H:%M:%S")
        data.set_index('Date', inplace=True)

        
        daily_costs = data.groupby(data.index.date)['Total_Cost'].sum()  
        daily_costs.index = pd.to_datetime(daily_costs.index)

        
        full_date_range = pd.date_range(start=daily_costs.index.min(), end=daily_costs.index.max())

        
        daily_costs = daily_costs.reindex(full_date_range)

       
        daily_costs.fillna(method='ffill', inplace=True)

       
        daily_costs.index.name = "Date"
        st.dataframe(daily_costs)
       
        st.sidebar.header("Forecasting Options")
        forecasting_model = st.sidebar.radio("Choose a Forecasting Model:", ["Exponential Smoothing", "ARIMA"])
        forecast_steps = st.sidebar.number_input("Forecast Steps (Days):", min_value=1, max_value=30, value=10)

       
        split_percentage = st.sidebar.slider("Training Data Percentage:", min_value=50, max_value=95, value=80)
        split_point = int(len(daily_costs) * split_percentage / 100)
        train = daily_costs[:split_point]
        test = daily_costs[split_point:]

        
        st.header("Sales Forecasting")
        
        if forecasting_model == "Exponential Smoothing":
            st.subheader("Exponential Smoothing Forecast")
            model_es = ExponentialSmoothing(train, trend="add", seasonal=None, damped_trend=True)
            fit_es = model_es.fit()
            forecast_es = fit_es.forecast(steps=forecast_steps)

            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(daily_costs, label='Historical', color='blue', linewidth=0.1)
            
            ax.plot(pd.date_range(start=daily_costs.index[-1] + pd.Timedelta(days=1), periods=forecast_steps), forecast_es, label='Forecast', color='orange', linewidth=0.7)
            
            ax.legend()
            ax.set_title("Exponential Smoothing Forecast")
            ax.set_ylabel("Total Cost")
            ax.set_xlabel("Date")
            st.pyplot(fig)

        elif forecasting_model == "ARIMA":
            st.subheader("ARIMA Forecast")
            model_arima = ARIMA(train, order=(1, 1, 1))
            fit_arima = model_arima.fit()
            forecast_arima = fit_arima.forecast(steps=forecast_steps)

           
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(daily_costs, label='Historical', color='blue', linewidth=0.1)
            ax.plot(pd.date_range(start=daily_costs.index[-1] + pd.Timedelta(days=1), periods=forecast_steps), 
                    forecast_arima, label='Forecast', color='green', linewidth=0.7)
            ax.legend()
            ax.set_title("ARIMA Forecast")
            ax.set_ylabel("Total Cost")
            ax.set_xlabel("Date")
            st.pyplot(fig)

      
        st.write(f"Training data from {train.index.min()} to {train.index.max()}")
        st.write(f"Testing data from {test.index.min()} to {test.index.max()}")

       
        if len(test) >= forecast_steps:
            test_forecast_comparison = test[:forecast_steps]
            st.write("True vs. Forecast for the first test steps:")
            st.write(pd.DataFrame({"True Values": test_forecast_comparison, 
                                "Forecasted Values": forecast_es if forecasting_model == "Exponential Smoothing" else forecast_arima}))
                 
            
    
    

if __name__ == '__main__':
    main()
