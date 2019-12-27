import pandas as pd

#split dataset in csv that contain 1000000 lines of data each
def split_data(PATHFILE):
    csvfile = open(PATHFILE,'r').readlines()
    filename = 1
    for i in range(len(csvfile)):

         if i%100000 == 0:
             open(str(filename) + '.csv', 'w+').writelines(csvfile[i:i+100000])
             filename += 1

#print csv
if __name__ == "__main__":

    split_data('/home/nick/Desktop/thesis/datasets/recsys-challenge-2015/processed/rsc15_train_full.txt')
    #df = pd.read_csv('/home/nick/Desktop/thesis/datasets/recsys-challenge-2015/processed/train_full_splitted/1.csv',delimiter='\t')

#print(df)