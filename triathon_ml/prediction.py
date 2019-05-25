import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

class prediction:

    # rating should be between 0 and 1
    def load_model_and_pred(self, model_name = 'model.pickle', rating = 0.9, wins = 7):
        
        try :
            # loading the saved model
            pickle_in = open('.././'+model_name, 'rb')
            regressor = pickle.load(pickle_in)
            y_pred = regressor.predict([[rating,wins]])
            y_pred = np.clip(y_pred, 0, 5)
            return y_pred

        except:
            return 'No model present in the current directory.'

    #trains the model on dataset
    def train_model(self, csv_file = 'lawers.csv', save_model_name = 'model.pickle'):
        
        try:
            # loading dataset
            dataset = pd.read_csv('../'+csv_file)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, 3].values
            regressor = LinearRegression()
            regressor.fit(X,y)
            pickle_out = open('../'+save_model_name, 'wb')
            pickle.dump(regressor, pickle_out)
            pickle_out.close()
            print(dir)
            return 'Done'

        except:
            return 'csv file is not present in the current directory.'

    # generate a dataset with random numbers
    def generate_a_dataset(self, qnt_of_rand_num = 20, dataset_name = 'lawers.csv'):
        
        #genarate random numbers in range [0,1)
        rating = np.random.random(qnt_of_rand_num)

        # generate a empty win list which will conatin numbers of wins individual advocate has
        win_list = []

        # generate an empty array to store randomly generated scores
        score_list = []

        # generate an empty array to store randomly generated timestamp difference
        time_diff = []

        # generate wins based on their ratings
        for i in rating:   
            if i > 0.7 :
                win = np.random.randint(6, 8)
                diff = np.random.randint(30, 100)
                time_diff.append(diff)
                win_list.append(win)
                continue
            if i > 0.5 and i < 0.7:
                win2 = np.random.randint(4, 6)
                diff = np.random.randint(50, 150)
                time_diff.append(diff)
                win_list.append(win2)
                continue
            elif i < 0.5 and i > 0.2:
                win3 = np.random.randint(2, 4)
                diff = np.random.randint(270, 320)
                time_diff.append(diff)
                win_list.append(win3)
                continue
            else :
                win4 = np.random.randint(0, 2)
                diff = np.random.randint(500, 1000)
                time_diff.append(diff)
                win_list.append(win4)
                continue

        df = pd.DataFrame()
        df['Ratings'] = rating
        df['Wins'] = win_list
        df['Time_diff'] = time_diff
        for i in win_list:
            if i >= 0 and i<=2:  
                rand_score = np.random.uniform(0, 1.5)
                score_list.append(rand_score)
                continue
            elif i >= 2 and i <= 4:
                rand_score = np.random.uniform(2, 3.2)
                score_list.append(rand_score)
                continue        
            elif i >= 4 and i <= 6:
                rand_score = np.random.uniform(3.5, 4)
                score_list.append(rand_score)
                continue             
            elif i >= 6 and i<= 7:
                rand_score = np.random.uniform(3.7, 5)
                score_list.append(rand_score)
                continue           
                
        df['Score'] = score_list
        df.to_csv(dataset_name, index=False, encoding='utf8')