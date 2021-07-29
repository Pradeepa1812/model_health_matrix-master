# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 19:04:23 2019

@author: ADMIN
"""

#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

basepath = os.path.dirname('__file__')
filepath = os.path.abspath("..")
datapath = "/Users/ibytecode/Downloads"
srcpath = filepath + "\src"
if filepath not in sys.path:
    sys.path.append(filepath)
if srcpath not in sys.path:
    sys.path.append(srcpath)
import copy
import pandas as pd
import numpy as np
#from classStruct.model import model
#from classStruct.cluster import cluster
from cvxopt import matrix, solvers
#from classStruct.pcr import pcr
from sklearn.decomposition import SparsePCA
import statsmodels.api as sm
import matplotlib.pyplot as plt

#####################################################################
##upload prepared sand and consumption data in csv formate
#####################################################################

#Output_col=['GCS (gm/cm2)','Compactability (no)','Active Clay (%)','Wet Tensile Strength(gm/cm2)','LOI (%)','Moisture (%)','Inert Fines (%)','Volatile Matter (%)','Permeability (no)','GFN/AFS (no)']
#Input_col=['Bentonite(MT)','LCA(MT)','Fresh Silica Sand(MT)','Return Sand(MT)','Water(MT)']
#Ratio_list=['Sand/Metal Ratio','Sand/Core Ratio','Core/Metal Ratio']
Ratio_list=['Sand/Metal Ratio','Core/Sand Ratio','Core/Metal Ratio']
additive_frac=list(['Bentonite_frac','Fresh Silica Sand_frac','LCA_frac','Return Sand_frac','Water_frac'])
input_data_block_total = pd.read_excel("input_munjal.xlsx")
output_data_block_total = pd.read_excel("output_munjal.xlsx")

#input_data_block_total = pd.read_csv(datapath+"\\input-MEI.csv")
#output_data_block_total = pd.read_csv(datapath+"\\output-MEI 11july.csv")
#structurecsv=pd.read_csv(datapath+"\\structure.csv",header=None)
#structurecsv=np.array(structurecsv)
Date_shift=input_data_block_total[['Date','Shift']]
#################################################################################
shifted_input=Ratio_list
shifted_prop_additivelist=list(output_data_block_total.columns)
shifted_prop_additivelist.extend(shifted_input)
inputFieldsList = list((input_data_block_total.columns))
inputFieldsList1 = list(shifted_prop_additivelist)
inputFieldsList2 = list(set(inputFieldsList)-set(inputFieldsList1)-set(['Date','Shift']))
outputFieldsList=list(output_data_block_total.columns)
inputFieldsList1.sort()
inputFieldsList2.sort()
outputFieldsList.sort()
input_columnlist=inputFieldsList1 +inputFieldsList2
inputFieldsList = inputFieldsList1 +inputFieldsList2
delayedoutputfieldlist=inputFieldsList1
input_data_block_total=input_data_block_total[input_columnlist]
data_len=len(input_data_block_total)
##################################################################################


data_lentrain=int(1*data_len)
print(data_lentrain)
between_data_points = data_len-data_lentrain
#i have commented here
#data_lentrain=int(0.10*data_len)

#xts = input_data_block_total.index>data_lentrain
#xts1= input_data_block_total.index<between_data_points
#xts_val = list(xts) and list(xts1)
#input_data_block=input_data_block_total.ix[xts_val]

#input_data_block=input_data_block_total.ix[input_data_block_total.index>data_lentrain and input_data_block_total.index<between_data_points]
#i have commented here
input_data_block=input_data_block_total.iloc[input_data_block_total.index<data_lentrain]
########
########
#input_data_block.index=range(len(input_data_block))
#output_data_block=output_data_block_total.ix[output_data_block_total.index>data_lentrain]
#i have commented here
output_data_block=output_data_block_total.iloc[output_data_block_total.index<data_lentrain]
####
output_data_block.index=range(len(output_data_block))
#######################################data for model testing
l=0
input_data_blocktest=input_data_block_total.iloc[input_data_block_total.index>l]
input_data_blocktest.index=range(len(input_data_blocktest))
output_data_blocktest=output_data_block_total.iloc[output_data_block_total.index>l]
output_data_blocktest.index=range(len(output_data_blocktest))
#    
Date_shift_test=Date_shift.iloc[input_data_block_total.index>0]
Date_shift_test.index=range(len(Date_shift_test))
#modeling_date=Date_shift.ix[xts_val]
#modeling_date=Date_shift.ix[input_data_block_total.index>data_lentrain and input_data_block_total.index<between_data_points]
#i have  commened here
modeling_date=Date_shift.iloc[input_data_block_total.index<data_lentrain]
#######
#    #
#    #####################################################################
#    ######################################################################
#    ######################### """Computes the maximum and minimum values of each column.
#    ###        
#    ####        :param data: Data whose minimum and maximum of each column need to be computed.
#    ####        :type data: DataFrame
#    ####        :return: Minimum, and Maximum values of each column of the given dataframe
#    ####        :rtype': DataFrame
#    ###        
#    ####################################################################
def computeBounds(data):      
        dataMinSeries = data.min().to_frame("min")
        dataMaxSeries = data.max().to_frame("max")
        dataMinMaxDataFrame = pd.concat([dataMinSeries,dataMaxSeries],axis=1,join='inner')
        return dataMinMaxDataFrame
dataInputMinMaxDataFrame=computeBounds(input_data_block)
dataOutputMinMaxDataFrame = computeBounds(output_data_block)
#
###############################################################################################
def NormalizeColumns(columnNameList,dataDF,dataMinMaxDataFrame):
        scaledData = dataDF.copy()
        for col in ((columnNameList)):
            den = dataMinMaxDataFrame["max"][col] - dataMinMaxDataFrame["min"][col]
            if (den == 0.0):
                scaledData[col] = 0.0
            else:
                scaledData[col] = (scaledData[col] - dataMinMaxDataFrame["min"][col] ) / den
        return scaledData
################################################################################################
#
scaledInputData=NormalizeColumns(input_columnlist,input_data_block,dataInputMinMaxDataFrame)
scaledoutputdata=NormalizeColumns(outputFieldsList,output_data_block,dataOutputMinMaxDataFrame)     
###
###################################################################
##################denormalize the column
###########################################################################################
def deNormalizeColumns(columnNameList,dataDF,dataMinMaxDataFrame):        
        unscaledData = dataDF.copy()
        for col in columnNameList:
            den = dataMinMaxDataFrame["max"][col] - dataMinMaxDataFrame["min"][col]
            if (den == 0.0):
                unscaledData[col] = dataMinMaxDataFrame["max"][col]
            else:
                unscaledData[col] = unscaledData[col] * den + dataMinMaxDataFrame["min"][col]
        return unscaledData        
########################################################################################
##
#unscaledData =deNormalizeColumns(input_columnlist,input_data_block,dataInputMinMaxDataFrame) 
##
#######################################################
###############make data mean centered
####################################################
#######################################################################################
###
def meanCenterData(dataDF):  
    DF = (dataDF).copy()
    meanSeries = DF.mean()
    DF = DF -  meanSeries
    return DF,meanSeries
datainput,inputMean = meanCenterData(input_data_block)
dataoutput,outputMean = meanCenterData(output_data_block)
datainput,inputMean = meanCenterData(scaledInputData)
dataoutput,outputMean = meanCenterData(scaledoutputdata)
######################################################################################
####
############################### find eigen vector and eigen value of input
########################################################################################
def findPCs(input_data_block,input_columnlist):
        scaledData=NormalizeColumns(input_columnlist,input_data_block,dataInputMinMaxDataFrame)
        datainput,inputMean = meanCenterData(scaledData)
        A = datainput[input_columnlist]
        A = A.values
        _,s,vh = np.linalg.svd(A)
        eVectors = vh.T
        eValues = s**2/(A.shape[0]-1.0)
        variancePercent = eValues/sum(eValues)
        varianceCaptured = 0.0
        percentageVarianceExplainedPC=0.95
        for i in range(len(eValues)):
            varianceCaptured = varianceCaptured + variancePercent[i]
            if (varianceCaptured >= percentageVarianceExplainedPC):
                break;
        retainedInputEigenVectors = eVectors[:,0:(i+1)]
        retainedInputEigenValues = eValues[0:(i+1)]
        return retainedInputEigenVectors,retainedInputEigenValues
retainedInputEigenVectors,retainedInputEigenValues=findPCs(input_data_block,input_columnlist)
##
###################################################################################
###################project data to reduced space
################################################################################
def projection2ReducedSpace(X):
        return (np.dot(X,retainedInputEigenVectors))
    
    
def estimateModelParameters(input_data_block,output_data_block,retainedInputEigenVectors):   
    scaledInputData=NormalizeColumns(input_columnlist,input_data_block,dataInputMinMaxDataFrame)
    scaledOutputData=NormalizeColumns(outputFieldsList,output_data_block,dataOutputMinMaxDataFrame)
    datainput,inputMean = meanCenterData(scaledInputData)
    dataoutput,outputMean = meanCenterData(scaledOutputData)
    inputScores = projection2ReducedSpace(datainput[input_columnlist].values)
    output =dataoutput[outputFieldsList].values
    m1 = np.dot(inputScores.T,inputScores)
    m1 = np.linalg.pinv(m1)
    m2 = np.dot(m1,inputScores.T)
    modelCoefficientsInReducedSpace = np.dot(m2,output)
    modelCoefficientsInOriginalSpace = np.dot(retainedInputEigenVectors,modelCoefficientsInReducedSpace)
    return modelCoefficientsInOriginalSpace



##
modelCoefficientsInOriginalSpace=estimateModelParameters(input_data_block,output_data_block,retainedInputEigenVectors)
coeff= pd.DataFrame(modelCoefficientsInOriginalSpace)
#coeff.to_csv("C:\Users\ADMIN\Desktop\Office work\MEI\Modelcoefficient.csv")

   
def predict(datainput):
    data = copy.deepcopy(datainput[inputFieldsList])
    data = data.reindex(inputFieldsList,axis=1)
    data = NormalizeColumns(input_columnlist,data,dataInputMinMaxDataFrame)
    data = data -inputMean[input_columnlist]
    prediction = np.dot(data.values,modelCoefficientsInOriginalSpace)
    prediction = pd.DataFrame(prediction,columns=outputFieldsList)
    prediction = prediction + outputMean[outputFieldsList]
    global pre_scaled
    pre_scaled=prediction
    prediction = deNormalizeColumns(outputFieldsList, prediction,dataOutputMinMaxDataFrame)
    return prediction
####################################################################################################################
    



####################################################################################################################
####################################################STATISTICS REPORT################################################
    

############################K-FOLD CROSS VALIDATION RMSE#############################################################
def StatReport(input_columnlist,input_data_block,outputFieldsList,output_data_block,dataInputMinMaxDataFrame,dataOutputMinMaxDataFrame,retainedInputEigenVectors) :
        scaledInputData=NormalizeColumns(input_columnlist,input_data_block,dataInputMinMaxDataFrame)
        scaledOutputData=NormalizeColumns(outputFieldsList,output_data_block,dataOutputMinMaxDataFrame)
        datainput,inputMean = meanCenterData(scaledInputData)
        dataoutput,outputMean = meanCenterData(scaledOutputData)
        inputScores = projection2ReducedSpace(datainput[input_columnlist].values)
        output =dataoutput[outputFieldsList].values
        inputScores=np.array(inputScores)
        num_folds = 10
        subset_size =int( len(inputScores)/num_folds)
        RMSE=[[],[],[],[],[],[],[],[],[],[]] # empy list should equal to number of folds
        for i in range(num_folds):
            testing_this_round_input = input_data_block[i*subset_size:][:subset_size]
            testing_this_round_output = output_data_block[i*subset_size:][:subset_size]
            training_this_round_input =  np.concatenate((np.array(inputScores[:i*subset_size]) , np.array(inputScores[(i+1)*subset_size:])),axis=0)
            training_this_round_output = np.concatenate((np.array(output[:i*subset_size]) , np.array(output[(i+1)*subset_size:])),axis=0)
            
            #model on train
            m1 = np.dot(training_this_round_input.T,training_this_round_input)
            m1 = np.linalg.pinv(m1)
            m2 = np.dot(m1,training_this_round_input.T)
            modelCoefficientsInReducedSpace = np.dot(m2,training_this_round_output)
            modelCoefficientsInOriginalSpace = np.dot(retainedInputEigenVectors,modelCoefficientsInReducedSpace)
            test=testing_this_round_input
            #pre=predict(test)
            ##############################################
            data = copy.deepcopy(test[inputFieldsList])
            data = data.reindex(inputFieldsList,axis=1)
            data = NormalizeColumns(input_columnlist,data,dataInputMinMaxDataFrame)
            data = data -inputMean[input_columnlist]
            prediction = np.dot(data.values,modelCoefficientsInOriginalSpace)
            prediction = pd.DataFrame(prediction,columns=outputFieldsList)
            prediction = prediction + outputMean[outputFieldsList]
            prediction = deNormalizeColumns(outputFieldsList, prediction,dataOutputMinMaxDataFrame)
            #################################################
            rms=[]
            from sklearn.metrics import mean_squared_error
            from math import sqrt
            for z in range(0, len(outputFieldsList)) :
              rms.append(sqrt(mean_squared_error(testing_this_round_output.iloc[:,z], prediction.iloc[:,z])))
           
            RMSE[i]=rms
        RMSE=np.array(RMSE)   
        RMSE_Report=pd.DataFrame(RMSE,columns=[outputFieldsList])   
        avg=np.average(RMSE_Report,axis=0)
        avg=np.array(avg.T)
        RMSE_Report_AVG= pd.DataFrame(avg,index=[outputFieldsList],columns=["Cross_Validated_RMSE"])
        ####################################################################################################################
        ####################################################################################################################
        
        
        ############################STAT REPORT FOR PCR MODEL###############################################################
        import statsmodels.api as smo
        import statsmodels.stats.stattools as sms
        Stat_Report = pd.DataFrame( index=["R-squared:","Adj. R-squared:","F-statistic:","Prob (F-statistic)","AIC:","BIC:","P_Values",
                                           "model_coef_original_space","Total mean squared error","Cross_validated_Sand_prop_RMSE",
                                            "RMSE_Sand_prop_after_optimize","RMSE_Sand_prop_after_optimize_scaled","RMSLE_Sand_prop_after_optimize",
                                            "Jarque-Bera (JB):","Prob(JB):","skew:","Kurtosis:","std err ","Durbin-Watson:"]
                                            ,columns=[outputFieldsList])
        
        for k in range(0,len(outputFieldsList)) :
            results = smo.OLS(output[:,k],inputScores).fit()
            #results.summary()
            jb=sms.jarque_bera(results.resid)
            db=sms.durbin_watson(results.resid)
            #print(db)
            para=np.dot(retainedInputEigenVectors,results.params)
            Stat_Report.iloc[0,k]=results.rsquared  ; Stat_Report.iloc[1,k]=results.rsquared_adj ;
            Stat_Report.iloc[2,k]=results.fvalue    ; Stat_Report.iloc[3,k]=results.f_pvalue     ;
            Stat_Report.iloc[4,k]=results.aic       ; Stat_Report.iloc[5,k]=results.bic          ; 
            Stat_Report.iloc[6,k]=results.pvalues   ; Stat_Report.iloc[7,k]=para                 ; 
            Stat_Report.iloc[8,k]=results.mse_total ; Stat_Report.iloc[13,k]=jb[0]
            Stat_Report.iloc[14,k]=jb[1]            ;Stat_Report.iloc[15,k]=jb[2]
            Stat_Report.iloc[16,k]=jb[3]            ;Stat_Report.iloc[17,k]=results.bse
            Stat_Report.iloc[18,k]=db 
        Stat_Report.iloc[9,:]=avg   # average of cross validated RMSE 
        
        
        return Stat_Report
#####################################################################################################################    
Stat_Report=StatReport(input_columnlist,input_data_block,outputFieldsList,output_data_block,dataInputMinMaxDataFrame,dataOutputMinMaxDataFrame,retainedInputEigenVectors)
#
#####################################################################################################################    

#
#
def optimize(optX,prevX):
##        #==========================================================modified on 20.09.2016
    optX1 = optX.reindex(outputFieldsList,axis = 0)
    #        #==========================================================
    prevX1 = prevX.reindex(delayedoutputfieldlist,axis = 0)#.as_matrix()
    #        
    optX1 = NormalizeColumns(outputFieldsList,optX1,dataOutputMinMaxDataFrame)
    ########print optX1
    prevX1 = NormalizeColumns(delayedoutputfieldlist,prevX1,dataInputMinMaxDataFrame)
    optX1 = optX1 - outputMean[outputFieldsList]
    prevX1 = prevX1 -inputMean[delayedoutputfieldlist]
    #        
    #                
    optX1 = optX1.values
    prevX1 = prevX1.values
    #        
    modelMatrix = modelCoefficientsInOriginalSpace.T
    nSandProp = len(delayedoutputfieldlist)
    nAll = len(input_columnlist)
    #        #nAdditives = len(inputFieldsList2)
    nAdditives =  nAll-nSandProp
    alpha = modelMatrix[:,:nSandProp]
    beta = modelMatrix[:,nSandProp:nAll]
    #beta=np.multiply(beta,structurecsv)          #  added changed
            ####'creating the weighting matrix'
            #===============================modified on 20.09.2016=====
    mSandProp = len(outputFieldsList)
    W = np.identity(mSandProp)
    #        #===================================
    P = 2.0 * np.dot(np.dot(beta.T,W),beta)
    #        
    q1temp = np.dot(prevX1,alpha.T)
    q1 = np.dot(np.dot(q1temp,W),beta)
    q2 = np.dot(np.dot(optX1,W),beta)
    q = 2.0 * (q1 -q2)
    G = np.vstack((-np.identity(nAdditives),np.identity(nAdditives)))
    #        
    h1 = np.zeros((nAdditives)) -inputMean[inputFieldsList2].reindex(inputFieldsList2, axis = 0)
    h2 = np.ones((nAdditives)) - inputMean[inputFieldsList2].reindex(inputFieldsList2, axis = 0)
    h = np.hstack((-h1.T,h2.T))
#    std=1.0
#    min_maxnew=pd.DataFrame([[0,19.9],[0,44,],[0,8.9],[0,0.273],[0,5.14],[0,3.27],[0,3.2],[0,3.3],[0,133],[0,12.2],[0,78.5],[0,16.1],[0,58.8],[0,44.9],[0.0127-std*0.000922,0.0127+std*0.000922],[0.96-std*0.00368,0.96+std*0.00368],[0.00503-std*0.000287,0.00503+std*0.000287],[0.0109-std*0.00149,0.0109+std*0.00149],[0.0104-std*0.00267,0.0104+std*0.00267]],index=inputFieldsList,columns=['min','max'])
#    delta = min_maxnew["max"] - min_maxnew["min"]
#    
    
    delta = dataInputMinMaxDataFrame["max"] - dataInputMinMaxDataFrame["min"]
    A = delta[inputFieldsList2].reindex(inputFieldsList2,axis = 0).values
    b = 1 - sum(inputMean[inputFieldsList2] * A)-sum(dataInputMinMaxDataFrame["min"][inputFieldsList2])
    #        
    P_matrix = matrix(P,tc='d')
    q_matrix = matrix(q,tc='d')
    G_matrix = matrix(G,tc='d') 
    h_matrix = matrix(h,tc='d')
    A_matrix = matrix(A,tc ='d')
    A_matrix = A_matrix.T
    b_matrix = matrix(b,tc='d')
    #        
    solvers.options["show_progress"] =False # Set True to see solver progress
    solvers.options["maxiters"] = 2000
    #        #===========================================================================
    #        ###########'Invoking optimizer'
    result = solvers.qp(P_matrix,q_matrix,G_matrix,h_matrix, A_matrix,b_matrix )
    rs = pd.Series(result['x'],index=inputFieldsList2)
    rs = rs + inputMean[inputFieldsList2]
    rs = deNormalizeColumns(inputFieldsList2, rs, dataInputMinMaxDataFrame.loc[inputFieldsList2])
    return rs,beta
additives=np.zeros((len(input_data_blocktest),len(inputFieldsList2)))
for i in range(len(input_data_blocktest)):
    pk_1=input_data_blocktest[delayedoutputfieldlist].iloc[i]
    prevX = pd.Series(pk_1,index=delayedoutputfieldlist)
    p_opt=[8.44,40.10,1609.01,55,3.91,4.58,3.28,142.96,479.64,42.23,3.51]
    optX = pd.Series(p_opt,index=outputFieldsList)
################################################################with new rejecttion
    optX=optX.reindex(index=outputFieldsList)        
######################################################################    
    rs,beta=optimize(optX,prevX)
    additives[i]=rs
predicted_additive=pd.DataFrame(additives,columns=inputFieldsList2)
actual_additive=np.matrix(input_data_blocktest[inputFieldsList2])
predicted_additive=np.matrix(pd.DataFrame(additives,columns=inputFieldsList2))
rmse_additive=np.zeros((len(inputFieldsList2),1))
for i in range(len(inputFieldsList2)):
    diff=actual_additive[:,i]-predicted_additive[:,i]
    diff=np.power(diff, 2)
    mean_diff=np.sqrt(np.mean(diff,0))
    rmse_additive[i]=mean_diff
rmse_additive=pd.DataFrame(rmse_additive.T,columns=inputFieldsList2) 
Mixture_capacity=3100.0
rmse_additves_kg=rmse_additive.multiply(Mixture_capacity)
addtive_prediclist=list(['Bentonite_predicted','FSS_predicted','LCA_predicted','Returnsand_predicted','water_predicted'])
predicted_additive=pd.DataFrame(additives,columns=addtive_prediclist)
predicted_additive.index=range(len(predicted_additive))
actual_additive=input_data_blocktest[inputFieldsList2]

############################################################################################    
######################  Additives in kg for group all
############################################################################################    
predicted_additive_column=[]
for i in range(len(addtive_prediclist)):
    modify_str_all=addtive_prediclist[i]+str('_Kg')
    predicted_additive_column.append(modify_str_all)
    predicted_additive.insert(1,modify_str_all,(predicted_additive[addtive_prediclist[i]]*Mixture_capacity/predicted_additive["Returnsand_predicted"]))    
predicted_additive_without_adjustment= predicted_additive.copy() 

##############################################################################################
############ actual additives in KG
####################################################################################
Group_input=['Bentonite','Fresh Silica Sand','LCA','Return Sand','Water']
predicted_col_list=[str("Adj_")+Group_input[i]+str("_pred") for i in range(len(Group_input))]
##########################################################################################    
####predicted additives for group
#steps predict additives
##convert into mass fraction
# then predict sand properties Output_col
##########################################################################################    
Group_info=list(['ALL'])
targets=pd.DataFrame([8.44,40.10,1609.01,55,3.91,4.58,3.28,142.96,479.64,42.23,3.51],columns=['ALL'])
targets.index=outputFieldsList
Handals={'Bentonite':'Active Clay (%)','Fresh Silica Sand':'None','LCA':'LOI (%)','Return Sand':'None','Water':'None'}
pred_coeff=pd.DataFrame([0.4,99999,0.6,99999,99999],columns=['ALL'])
#pred_coeff=pd.DataFrame([999999,999999,999999,999999,999999],columns=['ALL'])
pred_coeff.index=Group_input
Group_additives = {}
for i in range(len(Group_info)): 
    x=np.zeros((len(input_data_blocktest),len(Group_input)))
    for j in range(len(Group_input)):         
            if (Handals[Group_input[j]])=='None':
                x[:,j]= predicted_additive[predicted_additive_column[j]].values
                modify_string=Group_input[i]+str('pred')
            else:
                dev=targets[Group_info[i]][Handals[Group_input[j]]]-input_data_blocktest[Handals[Group_input[j]]]
                modify_string=Group_input[i]+str('pred')
                x[:,j]=(pd.DataFrame([(predicted_additive[predicted_additive_column[j]]+(dev/pred_coeff[Group_info[i]][Group_input[j]]))]) ).values
    x_new=pd.DataFrame(x,columns=Group_input)
    Group_additives[Group_info[i]] = x_new
    
    adjust_additive_pred=pd.DataFrame(x,columns=predicted_col_list)


###############################################################################################    

################################ converting it into mass frac

##############################################################################################
Group_additives_massfrac_sp = {}
for key in Group_additives:
      for j in range(len(Group_additives[key].columns)):
          Group_additives[key].insert(j,Group_input[j]+str('_frac'),Group_additives[key][Group_input[j]]/Group_additives[key].sum(axis=1))
      data_for_pred=pd.concat([input_data_blocktest[delayedoutputfieldlist],Group_additives[key][additive_frac]],axis=1)  
      data_for_pred=data_for_pred.reindex(inputFieldsList, axis=1)
      Group_additives_massfrac_sp[key]=data_for_pred
######################################################################################################          
#   
######################################################################################################
#
#
###########################################################################
##########Sand properties prediction
##############################################################################
def sand_pred(df):
    df = NormalizeColumns(inputFieldsList,df,dataInputMinMaxDataFrame)
    df = df -inputMean[inputFieldsList]
    df_new=df[inputFieldsList2]
    pred = np.dot(df_new.values,beta.T)
    pred = pd.DataFrame(pred,columns=outputFieldsList)
    pred = pred + outputMean[outputFieldsList]
    pred = deNormalizeColumns(outputFieldsList, pred,dataOutputMinMaxDataFrame)
    return pred
def sand_pred_group(df):
 pred_group=sand_pred(df)
 pred_all=sand_pred(Group_additives_massfrac_sp['ALL'])
 diff=pred_group-pred_all
 sand_prop_predicted_all=predict(Group_additives_massfrac_sp['ALL'])
 sand_pred_group=sand_prop_predicted_all+diff
 return sand_pred_group
########################################################################################
############ Sandproperties prediction for all groups
#########################################################################################
Sand_prop_pred_method1={}
for key in Group_additives_massfrac_sp:    
    pred=predict(Group_additives_massfrac_sp[key])
    Sand_prop_pred_method1[key]=pred

Sand_prop_pred_method2={}
for key in Group_additives_massfrac_sp:    
    pred_group=sand_pred_group(Group_additives_massfrac_sp[key])
    Sand_prop_pred_method2[key]=pred_group
####################################################################################
################# RMSE of sand properties
##################################################################################

def rmse_multivariate(Sand_prop_pred_method_dict,output_data_blocktest,target_df,scaledoutputdata,dataOutputMinMaxDataFrame,outputFieldsList):
    rmse_sand_prop_dict={}
    for key in Sand_prop_pred_method1:
        rmse_sand_prop=np.zeros((len(outputFieldsList),1))
        rmse_sand_prop_scale=np.zeros((len(outputFieldsList),1))
        rmse_sand_prop_log=np.zeros((len(outputFieldsList),1))
        global pre_scaled
        pre_scaled = np.matrix(pre_scaled)
        scaledoutputdata=np.matrix(output_data_blocktest)
        if key=="ALL":
            for i in range(len(outputFieldsList)):
                prediction = np.matrix(Sand_prop_pred_method_dict[key])
                output_data_blocktest=np.matrix(output_data_blocktest)
                diff=output_data_blocktest[:,i]-prediction[:,i]
                #Uncomment to generate residuals plot
                ######################RESIDUALS PLOT##########################
                #fig,ax = plt.subplots()
                #mean = [np.mean(diff)]*85
                #ax.plot(diff,label=outputFieldsList[i], marker='o')
                #ax.plot(mean)             
                #plt.ylabel('Residual')
                #plt.xlabel('Data_points')
                #ax.legend(loc='upper right')
                #plt.show()
                ############################################################## 
                ###############################################################
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                scaler.fit(output_data_blocktest)
                prediction_s=scaler.transform(prediction)
                output_data_blocktest_s=scaler.transform(output_data_blocktest)
                diff_scale=output_data_blocktest_s[:,i]-prediction_s[:,i]
                diff_scale=np.power(diff_scale, 2)
                mean_diff_scale=np.sqrt(np.mean(diff_scale,0))
                rmse_sand_prop_scale[i]=mean_diff_scale
               #######################LOGSCALED RMSE##########################
                prediction_log = np.log(np.matrix(Sand_prop_pred_method_dict[key]))
                output_data_blocktest_log=np.log(np.matrix(output_data_blocktest))
                diff_log=np.add(output_data_blocktest_log[:,i],1)-np.add(prediction_log[:,i],1)
                diff_log=np.power(diff_log, 2)
                mean_diff_log=np.sqrt(np.mean(diff_log,0))
                rmse_sand_prop_log[i]=mean_diff_log
            
                #########################UNSCALED RMSE#########################
                diff=np.power(diff, 2)
                mean_diff=np.sqrt(np.mean(diff,0))
                rmse_sand_prop[i]=mean_diff
                
            rmse_sand_prop=pd.DataFrame(rmse_sand_prop.T,columns=outputFieldsList) 
            rmse_sand_prop_scale=pd.DataFrame(rmse_sand_prop_scale.T,columns=outputFieldsList)
            rmse_sand_prop_log=pd.DataFrame(rmse_sand_prop_log.T,columns=outputFieldsList)
            rmse_sand_prop_dict[key]=rmse_sand_prop
        else:
            for i in range(len(outputFieldsList)):
                prediction = np.matrix(Sand_prop_pred_method_dict[key])
                output_data_blocktest=target_df[key]
                diff=output_data_blocktest[i]-prediction[:,i]
                diff=np.power(diff, 2)
                mean_diff=np.sqrt(np.mean(diff,0))
                rmse_sand_prop[i]=mean_diff
            rmse_sand_prop=pd.DataFrame(rmse_sand_prop.T,columns=outputFieldsList) 
            rmse_sand_prop_dict[key]=rmse_sand_prop
    return rmse_sand_prop_dict,rmse_sand_prop_scale,rmse_sand_prop_log
rmse_sand_prop,rmse_sand_prop_scale,rmse_sand_prop_log=rmse_multivariate(Sand_prop_pred_method1,output_data_blocktest,targets,scaledoutputdata,dataOutputMinMaxDataFrame,outputFieldsList)
#
###############################STORING RMSE VALUES IN Stat_Report
Stat_Report.iloc[10,:]=np.array(rmse_sand_prop["ALL"])
Stat_Report.iloc[11,:]=np.array(rmse_sand_prop_scale)
Stat_Report.iloc[12,:]=np.array(rmse_sand_prop_log)
#print(Stat_Report)
input_data_blocktest_copy=input_data_blocktest.copy()
acual_additive_column=[]
for i in range(len(additive_frac)):
    modify_str=Group_input[i]+str('_Kg')
    acual_additive_column.append(modify_str)
    input_data_blocktest.insert(1,modify_str,(input_data_blocktest[additive_frac[i]]*Mixture_capacity/input_data_blocktest["Return Sand_frac"]))

actual_predicted=pd.concat([Date_shift_test,input_data_blocktest[['Active Clay (%)','GCS (kg/cm2)','LOI (%)','Wet Tensile Strength (gm/cm2)','Permeability (no)']],input_data_blocktest[acual_additive_column],predicted_additive_without_adjustment[predicted_additive_column],adjust_additive_pred],axis=1)
stat_analy_list=list(set(actual_predicted.columns)-set(["Date","Shift"]))

actual_predicted_stat=actual_predicted[stat_analy_list].describe()

STATISTICS={"SAND_PROPERTIES":Stat_Report,"ADDITIVES":rmse_additive}


#uncomment for exporting stat_report as excel file
#Stat_Report.to_excel("stat_report.xlsx")
print(rmse_additive)
print(Stat_Report)