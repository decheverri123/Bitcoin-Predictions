# Bitcoin-Predictions
Use of simple Linear Regression to build a simple model to predict bitcoin Highs and Lows




```python
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
df = quandl.get("GDAX/USD", authtoken="qdT3g8Du2XKrNBSDnKCo")
days = [i for i in range(len(df))]
```

Let's get the average price per day:


```python
zipped = zip(df.High, df.Low)
w = 0
df['AverageDay'] = np.nan
for i in zipped:
    df['AverageDay'][w] = np.average(i)
    w += 1
```


```python
df = df[['Open','High','Low','AverageDay']]
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>AverageDay</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-02-09</th>
      <td>8180.00</td>
      <td>8700.0</td>
      <td>7750.00</td>
      <td>8225.000</td>
    </tr>
    <tr>
      <th>2018-02-10</th>
      <td>8568.00</td>
      <td>9090.0</td>
      <td>8155.00</td>
      <td>8622.500</td>
    </tr>
    <tr>
      <th>2018-02-11</th>
      <td>8560.00</td>
      <td>8595.0</td>
      <td>7851.00</td>
      <td>8223.000</td>
    </tr>
    <tr>
      <th>2018-02-12</th>
      <td>8298.24</td>
      <td>8883.0</td>
      <td>8032.65</td>
      <td>8457.825</td>
    </tr>
    <tr>
      <th>2018-02-13</th>
      <td>8867.49</td>
      <td>8950.0</td>
      <td>8393.10</td>
      <td>8671.550</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(days, df.AverageDay)
```




    [<matplotlib.lines.Line2D at 0x1f2b13e8940>]




![png](output_5_1.png)


We want to predict the High for the day using that day's opening price:


```python
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
```


```python
X_high = np.array(df.drop(['AverageDay','Low','High'], 1))
y_high = np.array(df['High'])
```


```python
X_train_high, X_test_high, y_train_high, y_test_high = cross_validation.train_test_split(X_high,
                                                                                         y_high,test_size=0.25)
```


```python
clf = LinearRegression()
clf.fit(X_train_high, y_train_high)
accuracy = clf.score(X_test_high, y_test_high)
accuracy
```




    0.99640262474357788



We seem to get a very accurate model. With more time, it would behoove us to look into these later:


```python
def predictHigh(openingPrice):
    return int(clf.predict([openingPrice][0]))
```


```python
predictHigh(8867.49)
```




    9384



Now let's predict a day's lowest price using it's opening price:


```python
X_low = np.array(df.drop(['Low','High','AverageDay'], 1))
y_low = np.array(df['Low'])
```


```python
X_train_low, X_test_low, y_train_low, y_test_low = cross_validation.train_test_split(X_low,
                                                                                         y_low,test_size=0.25)
```


```python
clfLow = LinearRegression()
clfLow.fit(X_train_low, y_train_low)
accuracy = clfLow.score(X_test_low, y_test_low)
accuracy
```




    0.99602748830953425



Again, a very high accuracy rating, perhaps it has to do with the low number of features:


```python
def predictLow(openingPrice):
    return int(clfLow.predict([openingPrice][0]))
```


```python
predictLow(8867.49)
```




    8329




```python
def PredictLowHigh(openingPrice):
    print("Today's Highest Price Should Be:", predictHigh(openingPrice))
    print("Today's Lowest Price Should Be:", predictLow(openingPrice))
    print("With an average of:", (predictHigh(openingPrice)+predictLow(openingPrice))/2)
```


```python
PredictLowHigh(8867)
```

    Today's Highest Price Should Be: 9384
    Today's Lowest Price Should Be: 8328
    With an average of: 8856.0
    


