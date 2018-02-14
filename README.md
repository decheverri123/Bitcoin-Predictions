# Bitcoin-Predictions
Use of simple Linear Regression to build a simple model to predict bitcoin Highs and Lows


{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 387,
   "metadata": {},
   "outputs": [],
   "source": [
    "import quandl\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 388,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = quandl.get(\"GDAX/USD\", authtoken=\"qdT3g8Du2XKrNBSDnKCo\")\n",
    "days = [i for i in range(len(df))]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's get the average price per day:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 389,
   "metadata": {},
   "outputs": [],
   "source": [
    "zipped = zip(df.High, df.Low)\n",
    "w = 0\n",
    "df['AverageDay'] = np.nan\n",
    "for i in zipped:\n",
    "    df['AverageDay'][w] = np.average(i)\n",
    "    w += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 390,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Open</th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "      <th>AverageDay</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2018-02-09</th>\n",
       "      <td>8180.00</td>\n",
       "      <td>8700.0</td>\n",
       "      <td>7750.00</td>\n",
       "      <td>8225.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-02-10</th>\n",
       "      <td>8568.00</td>\n",
       "      <td>9090.0</td>\n",
       "      <td>8155.00</td>\n",
       "      <td>8622.500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-02-11</th>\n",
       "      <td>8560.00</td>\n",
       "      <td>8595.0</td>\n",
       "      <td>7851.00</td>\n",
       "      <td>8223.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-02-12</th>\n",
       "      <td>8298.24</td>\n",
       "      <td>8883.0</td>\n",
       "      <td>8032.65</td>\n",
       "      <td>8457.825</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-02-13</th>\n",
       "      <td>8867.49</td>\n",
       "      <td>8950.0</td>\n",
       "      <td>8393.10</td>\n",
       "      <td>8671.550</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Open    High      Low  AverageDay\n",
       "Date                                            \n",
       "2018-02-09  8180.00  8700.0  7750.00    8225.000\n",
       "2018-02-10  8568.00  9090.0  8155.00    8622.500\n",
       "2018-02-11  8560.00  8595.0  7851.00    8223.000\n",
       "2018-02-12  8298.24  8883.0  8032.65    8457.825\n",
       "2018-02-13  8867.49  8950.0  8393.10    8671.550"
      ]
     },
     "execution_count": 390,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df[['Open','High','Low','AverageDay']]\n",
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 394,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1f2b13e8940>]"
      ]
     },
     "execution_count": 394,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1f2afa7d358>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(days, df.AverageDay)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We want to predict the High for the day using that day's opening price:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 395,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn import preprocessing\n",
    "from sklearn.cross_validation import train_test_split\n",
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 396,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_high = np.array(df.drop(['AverageDay','Low','High'], 1))\n",
    "y_high = np.array(df['High'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 397,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_high, X_test_high, y_train_high, y_test_high = cross_validation.train_test_split(X_high,\n",
    "                                                                                         y_high,test_size=0.25)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 398,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.99640262474357788"
      ]
     },
     "execution_count": 398,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf = LinearRegression()\n",
    "clf.fit(X_train_high, y_train_high)\n",
    "accuracy = clf.score(X_test_high, y_test_high)\n",
    "accuracy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We seem to get a very accurate model. With more time, it would behoove us to look into these later:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 399,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predictHigh(openingPrice):\n",
    "    return int(clf.predict([openingPrice][0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 400,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9384"
      ]
     },
     "execution_count": 400,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predictHigh(8867.49)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now let's predict a day's lowest price using it's opening price:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 401,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_low = np.array(df.drop(['Low','High','AverageDay'], 1))\n",
    "y_low = np.array(df['Low'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 402,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_low, X_test_low, y_train_low, y_test_low = cross_validation.train_test_split(X_low,\n",
    "                                                                                         y_low,test_size=0.25)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 403,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.99602748830953425"
      ]
     },
     "execution_count": 403,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clfLow = LinearRegression()\n",
    "clfLow.fit(X_train_low, y_train_low)\n",
    "accuracy = clfLow.score(X_test_low, y_test_low)\n",
    "accuracy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Again, a very high accuracy rating, perhaps it has to do with the low number of features:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 404,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predictLow(openingPrice):\n",
    "    return int(clfLow.predict([openingPrice][0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 405,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8329"
      ]
     },
     "execution_count": 405,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predictLow(8867.49)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 406,
   "metadata": {},
   "outputs": [],
   "source": [
    "def PredictLowHigh(openingPrice):\n",
    "    print(\"Today's Highest Price Should Be:\", predictHigh(openingPrice))\n",
    "    print(\"Today's Lowest Price Should Be:\", predictLow(openingPrice))\n",
    "    print(\"With an average of:\", (predictHigh(openingPrice)+predictLow(openingPrice))/2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 407,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Today's Highest Price Should Be: 9384\n",
      "Today's Lowest Price Should Be: 8328\n",
      "With an average of: 8856.0\n"
     ]
    }
   ],
   "source": [
    "PredictLowHigh(8867)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

