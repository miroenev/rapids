import pandas as pd
import numpy as np
import matplotlib.pylab as plt

'''
source data 1: kaggle survey 2017 
source data 2: kaggle kernels -- https://raw.githubusercontent.com/adgirish/kaggleScape
'''

def set_rcParams():
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'figure.figsize': (12,7)})
    plt.rcParams.update({'figure.subplot.top': .99})
    plt.rcParams.update({'figure.subplot.left': .01})
    plt.rcParams.update({'figure.subplot.right': .99})
    
def count_responces(dataframe, targetCol):
    optionList = []
    for iRowResponceList in dataframe[targetCol].dropna().values:
        optionList += iRowResponceList.split(',')
    
    return pd.Series(optionList).value_counts()

def bar_plot_question(dataframe, schemaSurveyDF, qNum, nTopResults = None, newIndex = None):
    targetCol = schemaSurveyDF['Column'][qNum]
    uniqueCounts = count_responces( dataframe, targetCol)
    
    if nTopResults is not None:
        uniqueCounts = uniqueCounts.head(nTopResults)
    
    if newIndex is not None:        
        uniqueCounts = uniqueCounts.reindex( newIndex )
    plt.figure()
    
    uniqueCounts.plot.bar(alpha=1, color='tab:purple', rot=90)  
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', color='silver', linewidth=2)# plt.grid(True)    
    plt.gca().xaxis.grid(False)
    
    plt.title(schemaSurveyDF['Question'][qNum].split('(')[0])

def plot_datascientist_time_breakdown(dataframe, schemaSurveyDF, 
                                      columnsOfInterest = [ 'TimeGatheringData', 'TimeVisualizing', 
                                                            'TimeFindingInsights', 'TimeModelBuilding', 
                                                            'TimeProduction' ]):
    
    print('\t ' + schemaSurveyDF['Question'][220].split('(')[0])
    for iQuestion in range(220,225):
        print( '\t - ' + schemaSurveyDF['Question'][iQuestion].split('-')[1] )

    # remove nan's / non-responses
    cleanedColValues = []
    for iCol in range(len(columnsOfInterest)):
        cleanedColValues += [ dataframe[columnsOfInterest[iCol]].dropna().values ]

    # box plot
    patchData = plt.boxplot(x=cleanedColValues,            
                            patch_artist=True, notch=True, medianprops=dict(color='black'),
                            labels= ['Gathering Data', 'Visualizing', 'Exploring/Finding-Insights', 
                                     'Model Building', 'Productionalizing'])
    
    # make each data category a unique color
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for iCol in range(len(columnsOfInterest)):
        plt.setp(patchData['boxes'][iCol], color=colors[iCol], linewidth='5')
        plt.setp(patchData['whiskers'][iCol*2], color=colors[iCol], linewidth='5')
        plt.setp(patchData['whiskers'][iCol*2+1], color=colors[iCol], linewidth='5')

    plt.grid(True, linestyle='--', color='silver', linewidth=2)
    plt.ylim((-5,100))
    plt.ylabel('percentage of time spent')
    plt.show()
    
def pie_plot_important_activities_2018(kaggleSurveyDF_2018, schemaSurveyDF_2018 ):
    responseCountsList = []
    responseOptionsTextList = []
    responseOptionsTextListAnnotated = ['Analyze and understand data',
                                        'Use ML services to improve product or wokflows',
                                        'Build/run data infrastructure',
                                        'Apply ML to new areas and build prototypes',
                                        'Research/advance state of the art in ML',
                                        'Other / None of these activities are important to my role']

    for iOption in range(1,8):
        responseOptionsTextList += [ kaggleSurveyDF_2018['Q11_Part_' + str(iOption)][0].split('-')[-1].strip() ]
        responseCountsList += [ kaggleSurveyDF_2018['Q11_Part_' + str(iOption)][1:].dropna().value_counts()[0] ]

    responseCountsList[-2] = responseCountsList[-1] + responseCountsList[-2]
    responseCountsList = responseCountsList[:-1]
    plt.figure()
    plt.pie( x = responseCountsList,
            labels = responseOptionsTextListAnnotated,
            startangle=90,
            autopct='%1.1f%%',
            wedgeprops = { 'linewidth' : 1 , 'edgecolor' : 'white'},
            colors=np.array(plt.get_cmap('tab20').colors)
        );
    plt.show()
 
    
def bar_plot_2018(kaggleSurveyDF_2018, schemaSurveyDF_2018, qNum = 17):
    valueCounts = kaggleSurveyDF_2018['Q'+str(qNum)][1:].dropna().value_counts()
    textOptions = list(kaggleSurveyDF_2018['Q'+str(qNum)].dropna().value_counts().keys())[:-1]
    plt.figure()
    
    plt.show()
    pd.DataFrame({'label':textOptions, 'values':valueCounts}).plot.bar(alpha=1, color='tab:purple', rot=90);  
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', color='silver', linewidth=2)# plt.grid(True)    
    plt.gca().xaxis.grid(False)
    plt.title(kaggleSurveyDF_2018['Q'+str(qNum)][0].split('-')[0].strip())
    