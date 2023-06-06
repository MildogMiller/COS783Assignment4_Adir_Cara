import os
import sys
from optparse import OptionParser
import pandas as pd
import pickle
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB #to build our suspect detection model
from sklearn.model_selection import train_test_split #to split our dataset into two sets
from sklearn.pipeline import Pipeline
from progress.bar import Bar

class main():
    def main(self):
        if len(sys.argv[1:]) == 0:
            print("No arguments passed, please use -h or --help for help")
        else:
            self.withArgs()
    
    def withArgs(self):
        if options.train!= None and options.trainout== None:
            print("Please use -k or --train-out to set the output directory of the new trained model when training a new model")
            sys.exit()
        if options.train!= None and options.trainout!= None:
            # Check if input csv is in the correct format and exists
            if not options.train.endswith(".csv"):
                print("Please use a csv file for training")
                sys.exit()
            if not os.path.isfile(options.train):
                print('File does not exist')
                sys.exit()
            if os.path.isfile(options.trainout):
                check=input("Output already exists, overwrite? y/n: ")
                if check=="y" or check=="Y":
                    os.remove(options.trainout)
                else:
                    sys.exit()
            pipe=self.trainModelPipeline(options.train)
            self.savemodel(pipe,options.trainout)
            print("Thank you")
            sys.exit()
        if options.model== None:
            print("Please use -m or --model to set the model")
            sys.exit()
        if not os.path.isfile(options.model):
            print('Model does not exist')
            sys.exit()
        if options.output== None:
            print("Please use -o or --output to set the output file")
            sys.exit()
        if options.model!= None and (options.inputtxt== None and options.inputcsv== None):
            print("Please use (-i or --input-text) or (-c or --input-csv) to set the input file when using a model")
            sys.exit()
        if options.model!= None and (options.inputtxt!= None and options.inputcsv!= None):
            print("Please use either (-i or --input-text) or (-c or --input-csv) to set the input file when using a model")
            sys.exit()
        if options.model!= None and options.inputtxt!= None:
            if not os.path.isfile(options.inputtxt):
                print('Input file does not exist')
                sys.exit()
            if os.path.isfile(options.output):
                check=input("Output already exists, overwrite? y/n: ")
                if check=="y" or check=="Y":
                    os.remove(options.output)
                else:
                    sys.exit()
            self.runmodelTxt(options.model,options.inputtxt,options.output)
            print("Results saved, thank you")
            sys.exit()
        if options.model!= None and options.inputcsv!= None:
            if not os.path.isfile(options.inputcsv):
                print('Input file does not exist')
                sys.exit()
            if os.path.isfile(options.output):
                check=input("Output already exists, overwrite? y/n: ")
                if check=="y" or check=="Y":
                    os.remove(options.output)
                else:
                    sys.exit()
            self.runmodelCsv(options.model,options.inputcsv,options.output)
            print("Results saved, thank you")
            sys.exit()

    def runmodelTxt(self,modelname,inputtxt,output):
        # open output file for writing
        pipe=self.loadmodel(modelname)
        towritearr=[]
        suspectcount=0
        notsuspectcount=0
        linecount=len(open(inputtxt).readlines())
        with open(inputtxt, 'rb') as f:
            with Bar('Processing results',fill='@',suffix='%(percent).1f%% - %(eta)ds',max=linecount) as bar:
                for line in f:
                    pred = pipe.predict([line])
                    if pred==1:
                        suspectcount+=1
                        towritearr.append(line)
                    else:
                        notsuspectcount+=1
                    bar.next()
        with open(output, 'w',newline='') as f:
            with Bar('Writing results',fill='@',suffix='%(percent).1f%% - %(eta)ds',max=len(towritearr)) as bar:
                f.write("Suspect count: "+str(suspectcount)+"\n")
                f.write("Not suspect count: "+str(notsuspectcount)+"\n")
                f.write("Suspect emails\n")
                f.write("----------------------------------------------------------\n\n")
                for line in towritearr:
                    f.write(line.decode('utf-8'))
                    bar.next()

    def runmodelCsv(self,modelname,inputcsv,output):
        # open output file for writing
        pipe=self.loadmodel(modelname)
        towritearr=[]
        suspectcount=0
        notsuspectcount=0
        linecount=len(open(inputcsv).readlines())
        with open(inputcsv, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            with Bar('Processing results',fill='@',suffix='%(percent).1f%% - %(eta)ds',max=linecount-1) as bar:
                for row in reader:
                    pred = pipe.predict([row[1]])
                    if pred==1:
                        suspectcount+=1
                        towritearr.append(row)
                    else:
                        notsuspectcount+=1
                    bar.next()

        with open(output, 'w',newline='') as f:
            with Bar('Writing results',fill='@',suffix='%(percent).1f%% - %(eta)ds',max=len(towritearr)) as bar:
                f.write("Suspect count: "+str(suspectcount)+"\n")
                f.write("Not suspect count: "+str(notsuspectcount)+"\n")
                f.write("Suspect emails\n")
                f.write("----------------------------------------------------------\n\n")
                if options.savefull==True:
                    for line in towritearr:
                        # check if not last line
                        f.write(line[0]+","+line[1]+"\n")
                else:
                    for line in towritearr:
                        # check if not last line
                        if line!=towritearr[-1]:
                            f.write(line[0]+",")
                        else:
                            f.write(line[0])
                        bar.next()  

                

    def trainModelPipeline(self,traincsv):
        df = pd.read_csv(traincsv)
        df_data = df[["text","suspect"]]
        df_x = df_data['text']
        df_y = df_data['suspect']
        X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.30, random_state=42)
        pipe = Pipeline([('vectorizer', CountVectorizer()),('classifier', MultinomialNB())])
        pipe.fit(X_train,y_train)
        print("Model trained")
        print("Accuracy of Model",pipe.score(X_test,y_test)*100,"%")
        return pipe
        

    def savemodel(self,pipe,modelname):
        pickle.dump(pipe, open(modelname, 'wb'))
        print("Model saved as",modelname)

    def loadmodel(self,modelname):
        pipe = pickle.load(open(modelname, 'rb'))
        return pipe


if __name__ == "__main__":
    parser = OptionParser()
    # help option
    parser.add_option("-t", "--train", dest="train",help="Train a new model for this run using given CSV (csv format ['text','suspect'])", metavar="CSV")
    parser.add_option("-k", "--train-out", dest="trainout",help="Set the output directory/name of the new trained model (requires the -t command to be used)", metavar="CSV")
    parser.add_option("-m", "--model", dest="model",help="Use a model for this run", metavar="MODEL")
    parser.add_option("-o", "--output", dest="output",help="Set the output directory of the report", metavar="OUTPUT")
    parser.add_option("-i", "--input-text", dest="inputtxt",help="Set the input file as a txt file, with each line being a new email", metavar="INPUT")
    parser.add_option("-c", "--input-csv", dest="inputcsv",help="Set the input file as a csv file ['id','email']", metavar="INPUT")
    parser.add_option("--save-full", dest="savefull",help="Include both the email and id in the output",default=False,action="store_true")    


    (options, args) = parser.parse_args()

    main().main()