from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM

import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#make the data set

def create_window(data, window_size = 1):#function to create a windowed data set.    
    data_s = data.copy()
    for i in range(window_size):
        data = pandas.concat([data, data_s.shift(-(i + 1))], 
                            axis = 1)
        
    data.dropna(axis=0, inplace=True)
    return(data)

# read the data from csv file

train = pandas.read_csv('Tokyo__weather.csv')

# lagsp has 7 misssing values in train data and rest is tha in all entries and also drop un-necessary variable
#remove Date and rainfall because we want the other three temperatures to be predicted
train = train.drop(['Date', 'rainfall'], axis = 1)

#construct dataframe from the necessary values
train = pandas.DataFrame(train.values)
dimsize = 5
X_train = []
Y_train = []
X_test = []


#construct time series data set

series = create_window(train, dimsize)
X_train = series.iloc [:, :-3]
Y_train = series.iloc [:, [-3, -2, -1]]
X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
X_test = X_train

d = 0.2

#Construct LSTM
rnn_model = Sequential() 
#LSTM input layer with 200 units.
rnn_model.add(LSTM(input_shape= (dimsize*3,1),units= dimsize*3, return_sequences=True))
#Dropout layer to prevent overfitting
rnn_model.add(Dropout(d))
#A second LSTM layer with 256 units.
rnn_model.add(LSTM(256))
#A further Dropout layer.
rnn_model.add(Dropout(d))
#Three Dense layers to produce a three outputs( Max, Min, Average temperatures).

rnn_model.add(Dense(3))
#Use MSE as loss function.
rnn_model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])

rnn_model.fit(
    X_train, #x data
    Y_train, #y data
    batch_size=10, #print the result every batch_size times
    epochs=10) #total number of epochs

# make predictions according to the data
newY = rnn_model.predict(X_train)

result_data = []
N =  len ( newY)
newY = list(newY)
# This is the list to predict the value
resultY = []

year = 2019
month = 1
day = 1

#This is the month days. This is necessary to calculate the date
mon = [0, 31, 28, 31, 30,31,30, 31, 31, 30, 31, 30, 31]

printY = ['Date', 'MaxT', 'MinT', 'AverageT']
resultY.append ( printY)


#Predict the temperature from January 2 to December 31
for dd in range ( 364):
    day += 1
    if day>mon[month]:
        day = 1
        month += 1
    #get the string of the day
    strDate = str(year) +"."+str(month)+"."+str(day)

    #create new testX according to the previous time series data
    if dd>=365-dimsize:
        newX = []
        for i in range ( -dimsize, 0):
            for j  in range ( 3):
                newX.append ( [newY[i][j]])
        testX = [[newX]]
    else:
        testX = []
        for ele in X_test[dd]:
            testX.append ( ele)
        testX = [[testX]]
    testY =  rnn_model.predict(testX)
    printY = []
    #insert date, maximum temperature, minimum temperature and average temperature
    printY.append ( strDate)
    printY.append ( testY[0][0])
    printY.append ( testY[0][1])
    printY.append ( testY[0][2])
    newY.append ( list(testY[0]))
    resultY.append ( printY)
#print the data to csv file
dfrlt = pandas.DataFrame ( resultY)
dfrlt.to_csv ( 'result.csv',index = False, header = False)


#draw graph


def draw(aX, aY, pY, subtitle):
    fig = plt.figure(num=None, figsize=(20, 5), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot (1,1,1)
    mtitle = 'Compare Weather Forecast vs Actual ('
    mtitle += subtitle
    mtitle += ')'
    ax.title.set_text (mtitle)
    plt.xlabel ("Time Periods")
    plt.ylabel ( "Temperature")
    plt.scatter ( aX, pY, color = 'blue')
    plt.scatter ( aX, aY, color = 'red')
    red_patch = mpatches.Patch ( color = 'red', label = 'Forecast')
    blue_patch = mpatches.Patch ( color = 'blue', label = 'Actual')
    plt.legend ( loc = 'upper left', handles = [red_patch, blue_patch])
    filename = subtitle + '.png'
    plt.savefig ( filename)
    plt.show()
    
dfa = pandas.read_excel ( "weather3.xlsx" )
dfp = pandas.read_csv ( "result.csv")
aN = len( dfa)
aX = []
aYmax = []; aYmin = []; aYave = []
pYmax = []; pYmin = []; pYave = []

for i in range ( 7, aN):
    aX.append ( i-7)
    aYmax.append ( float(dfa['Column5'][i]))
    pYmax.append ( dfp['MaxT'][i-7])
    aYmin.append ( float(dfa['Column8'][i]))
    pYmin.append ( dfp['MinT'][i-7])
    aYave.append ( float(dfa['Column11'][i]))
    pYave.append ( dfp['AverageT'][i-7])
draw ( aX, aYmax, pYmax, 'Maximum Temperature')
draw ( aX, aYmin, pYmin, 'Miniimum Temperature')
draw ( aX, aYave, pYave, 'Average Temperature')
