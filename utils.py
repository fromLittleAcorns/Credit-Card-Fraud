#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:39:31 2017

@author: johnrichmond
"""
import numpy as np
import pandas as pd

def value_to_class(my_value, my_dict):
    return my_dict.get(my_value, 4)


class UnitRange():
    @staticmethod
    def unitRange(raw):
        raw_max=np.max(raw,axis=0)
        raw_min=np.min(raw,axis=0)
        processed=(raw-raw_min)/(raw_max-raw_min)
        return processed, raw_min, raw_max


class InvertUnitRange():
    @staticmethod
    def unitRange(data,adj_min,adj_max):
        range=np.subtract(adj_max, adj_min)
        processed = np.add(np.multiply(data,range),adj_min)
        return processed


class Normalize():
    @staticmethod
    def norm(raw):
        cases=raw.shape[0]
        features=raw.shape[1]
        raw_mean=np.mean(raw, axis=0)
        raw_std=np.std(raw, axis=0)
        processed=((raw-raw_mean)/raw_std)
        return processed, raw_mean, raw_std


class Catagorize():
    @staticmethod
    def classify(class_dic, targets):
        cases=targets.shape[0]
        catagorys=np.zeros([cases],dtype='long')
        no_cls=len(class_dic)
        for i in range(cases):
            for cls in range(no_cls):
                if targets[i]<value_to_class(cls,class_dic):
                    catagorys[i]=cls
                    break
                catagorys[i]=no_cls
        return catagorys


class ShuffleArrays():
    def __init__(self,length):
        self.length = length
        self.P = np.random.permutation(self.length)
        
    def unison_shuffled_copies(self,array):
        return array[self.P]


class ShuffleArrays2():
    @staticmethod
    def unison_shuffled_copies(*args):
        no_args=len(args)
        if no_args>1:
            for i in range(1,no_args):
                assert len(args[i])==len(args[0])
        p = np.random.permutation(len(args[0]))
        shuffled=[]
        for arg in args:
            shuffled.append(arg[p])
        return shuffled


class ShuffleArrays3():
    @staticmethod
    def unison_shuffled_copies(*args):
        no_args=len(args)
        if no_args>1:
            for i in range(1,no_args):
                assert len(args[i])==len(args[0])
        p = np.random.permutation(len(args[0]))
        return [arg[p] for arg in args]


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def add_quarter_and_normalize(dataset, all_features=False):
    # Function to create four one hot encodings for the quarters of the day based upon the time value in the dataset
    # the time value is in seconds from the start of the log

    # Will convert time to quarters using a function and Pandas data frame operation
    # Add column for quarter as numbers
    dataset.insert(1, 'qtr_num', 0)
    # Add columns for each quarter of a day after the qtr_num column
    for i in range(4, 0, -1):
        dataset.insert(2, 'Q' + str(i), 0)

    def to_quarter(row):
        seconds_in_quarter = 21600
        quarter = int(row["Time"] / seconds_in_quarter) % 4
        row['qtr_num']=quarter+1
        return row

    dataset = dataset.apply(to_quarter, axis=1)

    # Assign values to quarters columns
    for qtr in range(1, 5):
        dataset.loc[:,'Q' + str(qtr)] = np.where(dataset['qtr_num'] == qtr, 1, 0)

    # will now normalize selected columns.  Will first create copy of the amount column for future use
    ss_amount = StandardScaler()
    dataset['Amt_To_Keep'] = dataset['Amount']

    #dataset.loc['Amount'] = ss_amount.fit_transform(dataset.loc['Amount'].values.reshape(-1, 1))
    ss_amount.fit(dataset.loc[:, 'Amount'].values.reshape(-1,1))
    tmp=ss_amount.transform(dataset.loc[:,'Amount'].values.reshape(-1,1))
    dataset['Amount']=tmp
    # If required to normalise the other features then do so now
    if all_features==True:
        ss_vFeatures=StandardScaler(copy=False, with_mean=True, with_std=True)
        vFeatures=[]
        for i in range(1,29):
            vFeatures.append('V'+str(i))
        ss_vFeatures.fit(dataset.loc[:,vFeatures])
        tmp=ss_vFeatures.transform(dataset.loc[:,vFeatures])
        dataset.loc[:,vFeatures]=tmp

    return dataset


def add_quarter_and_unitRange(dataset):
    # Function to create four one hot encodings for the quarters of the day based upon the time value in the dataset
    # the time value is in seconds from the start of the log

    # Will convert time to quarters using a function and Pandas data frame operation
    # Add column for quarter as numbers
    dataset.insert(1, 'qtr_num', 0)
    # Add columns for each quarter of a day after the qtr_num column
    for i in range(4, 0, -1):
        dataset.insert(2, 'Q' + str(i), 0)

    def to_quarter(row):
        seconds_in_quarter = 21600
        quarter = int(row["Time"] / seconds_in_quarter) + 1
        return quarter%4+1

    dataset['qtr_num'] = dataset.apply(to_quarter, axis=1)

    # Assign values to quarters columns
    for qtr in range(1, 5):
        dataset['Q' + str(qtr)] = np.where(dataset['qtr_num'] == qtr, 1, 0)

    # limit range to 0-1 for all columns
    ss_amount = MinMaxScaler()
    dataset['Amt_To_Keep'] = dataset['Amount']

    #dataset[dataset.columns] = ss_amount.fit_transform(dataset[dataset.columns])
    #dataset['Amount'] = ss_amount.fit_transform(dataset['Amount'].values.reshape(-1, 1))
    ss_amount.fit(dataset.loc[:, 'Amount'])
    tmp=ss_amount.transform(dataset.loc[:,'Amount'])
    dataset['Amount']=tmp
    #    dataset.drop(['Time'], axis=1)
    #    dataset.drop(['qtr_num'], axis=1)

    return dataset



