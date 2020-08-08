from __future__ import print_function

import pandas
import numpy
import sys
import matplotlib

import thinkstats2


def ReadFemResp(dct_file='2015_2017_FemRespSetup.dct',
                dat_file='2015_2017_FemRespData.dat',
                nrows=None):
    dct = thinkstats2.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, nrows=nrows)
    CleanFemResp(df)
    df.to_csv(r'C:\Users\Anna\Documents\GitHub\dsc530\Final project\responsedata.csv')
    return df

def CleanFemResp(df):
    """Recodes variables from the respondent frame.

    df: DataFrame
    """
    pass



ReadFemResp()



