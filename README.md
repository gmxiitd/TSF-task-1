{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "778b2dbc",
   "metadata": {},
   "source": [
    "# Name- Ghulam Mustafa\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b30db2bb",
   "metadata": {},
   "source": [
    "## Task1- Prediction using supervised machine learning\n",
    "In this task we have to predict the % score of student based on the no of hours studied. The task has 2 variables where the feature is the no of hours and target value is the % score.  This can be solve using simple linear regression."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bef38705",
   "metadata": {},
   "source": [
    "## Steps -\n",
    "1- Importing libraries.\n",
    "2- Data Visualization.\n",
    "3- Data Preaparation.\n",
    "4- Training the Alogarithms\n",
    "5- Visualizing the model.\n",
    "6- Making the prediction.\n",
    "7- Evaluation the model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d563d799",
   "metadata": {},
   "source": [
    "## Step1- Importing the libraries and dataset.\n",
    "In this step, we will import the dataset through the link with the help of pandas library and then we will observe the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5ffebe79",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Importing the libraries.\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e3013d84",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Hours  Scores\n",
      "0     2.5      21\n",
      "1     5.1      47\n",
      "2     3.2      27\n",
      "3     8.5      75\n",
      "4     3.5      30\n",
      "5     1.5      20\n",
      "6     9.2      88\n",
      "7     5.5      60\n",
      "8     8.3      81\n",
      "9     2.7      25\n",
      "10    7.7      85\n",
      "11    5.9      62\n",
      "12    4.5      41\n",
      "13    3.3      42\n",
      "14    1.1      17\n",
      "15    8.9      95\n",
      "16    2.5      30\n",
      "17    1.9      24\n",
      "18    6.1      67\n",
      "19    7.4      69\n",
      "20    2.7      30\n",
      "21    4.8      54\n",
      "22    3.8      35\n",
      "23    6.9      76\n",
      "24    7.8      86\n"
     ]
    }
   ],
   "source": [
    "#Resading data\n",
    "url='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'\n",
    "data = pd.read_csv(url)\n",
    "print( data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "690cff75",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25 entries, 0 to 24\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Hours   25 non-null     float64\n",
      " 1   Scores  25 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 528.0 bytes\n"
     ]
    }
   ],
   "source": [
    "data.info ()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a1f800f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(25, 2)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "71c2a904",
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
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>25.000000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.012000</td>\n",
       "      <td>51.480000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.525094</td>\n",
       "      <td>25.286887</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.100000</td>\n",
       "      <td>17.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.700000</td>\n",
       "      <td>30.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4.800000</td>\n",
       "      <td>47.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.400000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>9.200000</td>\n",
       "      <td>95.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Hours     Scores\n",
       "count  25.000000  25.000000\n",
       "mean    5.012000  51.480000\n",
       "std     2.525094  25.286887\n",
       "min     1.100000  17.000000\n",
       "25%     2.700000  30.000000\n",
       "50%     4.800000  47.000000\n",
       "75%     7.400000  75.000000\n",
       "max     9.200000  95.000000"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "bd03f9e9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Hours     0\n",
       "Scores    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "51fe7ba6",
   "metadata": {},
   "source": [
    "## Step-2 Data Visualization\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2a005ca4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<function matplotlib.pyplot.show(close=None, block=None)>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": 
 "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plotting the distribution of scores \n",
    "data.plot(x=\"Hours\",y='Scores',style=\"*\",color='blue',markersize=10)\n",
    "plt.title ('hours vs scores')\n",
    "plt.xlabel('hours studied')\n",
    "plt.ylabel('score received')\n",
    "plt.grid()\n",
    "plt.show\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bb55bfb7",
   "metadata": {},
   "source": [
    "so from the graph above ,it is clear that there is linear relationship between 'Hours studied' and 'scores received'. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "47fdae80",
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
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Hours</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.976191</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Scores</th>\n",
       "      <td>0.976191</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Hours    Scores\n",
       "Hours   1.000000  0.976191\n",
       "Scores  0.976191  1.000000"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Correlation\n",
    "data.corr()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0b5b7550",
   "metadata": {},
   "source": [
    "## Step 3- data preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f20558d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# independent and dependent features using iloc function\n",
    "x=data.iloc[:,:1].values\n",
    "y=data.iloc[:,1:].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "881bb924",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2.5],\n",
       "       [5.1],\n",
       "       [3.2],\n",
       "       [8.5],\n",
       "       [3.5],\n",
       "       [1.5],\n",
       "       [9.2],\n",
       "       [5.5],\n",
       "       [8.3],\n",
       "       [2.7],\n",
       "       [7.7],\n",
       "       [5.9],\n",
       "       [4.5],\n",
       "       [3.3],\n",
       "       [1.1],\n",
       "       [8.9],\n",
       "       [2.5],\n",
       "       [1.9],\n",
       "       [6.1],\n",
       "       [7.4],\n",
       "       [2.7],\n",
       "       [4.8],\n",
       "       [3.8],\n",
       "       [6.9],\n",
       "       [7.8]])"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    " x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ee6ed9ea",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[21],\n",
       "       [47],\n",
       "       [27],\n",
       "       [75],\n",
       "       [30],\n",
       "       [20],\n",
       "       [88],\n",
       "       [60],\n",
       "       [81],\n",
       "       [25],\n",
       "       [85],\n",
       "       [62],\n",
       "       [41],\n",
       "       [42],\n",
       "       [17],\n",
       "       [95],\n",
       "       [30],\n",
       "       [24],\n",
       "       [67],\n",
       "       [69],\n",
       "       [30],\n",
       "       [54],\n",
       "       [35],\n",
       "       [76],\n",
       "       [86]], dtype=int64)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "d24b4631",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting into training and test sets \n",
    "from sklearn.model_selection import train_test_split\n",
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a053ed5",
   "metadata": {},
   "source": [
    "## Step-4 Training the alogarithm "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "3784117a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "model=LinearRegression()\n",
    "model.fit(x_train,y_train)\n",
    "\n",
    "         "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "33d891ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Making Predictions\n",
    "y_pred=model.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "b0f9be37",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[79.35322195],\n",
       "       [87.42041185],\n",
       "       [94.47920301],\n",
       "       [37.00047496],\n",
       "       [28.93328505]])"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "6da9dcac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[85],\n",
       "       [75],\n",
       "       [88],\n",
       "       [30],\n",
       "       [30]], dtype=int64)"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "564ce2dc",
   "metadata": {},
   "source": [
    "## Step-5 Visualizing the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "6e8b872e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# plotting the Regression line\n",
    "line= model.coef_*x+model.intercept_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "9f660f08",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<function matplotlib.pyplot.show(close=None, block=None)>"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": 
