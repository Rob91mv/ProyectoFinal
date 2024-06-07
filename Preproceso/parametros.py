import pandas as pd
import os, sys
sys.path.append(os.getcwd())

#Loading data

contract = pd.read_csv('Users/rmmniv/Library/Mobile%20Documents/com~apple~CloudDocs/Data%20Science/SPRINT%2017%20-%20Proyecto%20Final/ProyectoFinal/contract.csv')
personal = pd.read_csv('Users/rmmniv/Library/Mobile%20Documents/com~apple~CloudDocs/Data%20Science/SPRINT%2017%20-%20Proyecto%20Final/ProyectoFinal/final_provider/personal.csv')
internet = pd.read_csv('/datasets/final_provider/internet.csv')
phone = pd.read_csv('/datasets/final_provider/phone.csv')