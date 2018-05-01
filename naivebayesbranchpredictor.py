import re
import copy
import matplotlib.pyplot as plt
import numpy as np


## NB = NAIVE BAYES
## LR= LOGISTIC REGRESSION
## AT = ALWAYS TAKEN
## NT = NEVER TAKEN

### ACCURACY LIST FOR GRAPHING
accuracy_list_nb = []
accuracy_list_at = []
accuracy_list_lr = []
accuracy_list_nt = []

### PRECISION LIST FOR GRAPHING
precision_list_nb = []
precision_list_at = []
precision_list_lr = []
precision_list_nt = []

### RECALL LIST FOR GRAPHING
recall_list_nb = []
recall_list_at = []
recall_list_lr = []
recall_list_nt = []

### F1 MEASURE LIST FOR GRAPHING
f1_measure_list_nb = []
f1_measure_list_at = []
f1_measure_list_lr = []
f1_measure_list_nt = []

###WEIGHT VECTOR FOR LOGISTIC REGRESSION
w = []
def intToBinaryStr(i):
    if i == 0:
        return "0"
    s = ''
    while i:
        if i & 1 == 1:
            s = "1" + s
        else:
            s = "0" + s
        i >>= 1
    return s

####################################
### TRANSFORM INTEGER TO BINARY FOR TABLE
#####################################
def getMSB(number):
    intStr = intToBinaryStr(number)
    return intStr[0]


#####################################
### PARSE THE TRACE FILE
#####################################
def parse(fparam):
    content = []
    filtered = []
    bh = []

    ### TO KNOW THAT THE PROGRAM ISN'T FROZEN, PRINT
    print("processing")
    with open(fparam) as f:
        for line in f:
            # SEPERATES THE COLUMNS
            temp = line
            temp1 = re.sub("\s+", " ", temp)
            sp = temp1.split(' ')
            entry = sp[6]
            # READS IN THE N/T INTO BRANCH HISTORY
            if (entry == 'N'):
                bh.append(0)
            if (entry == 'T'):
                bh.append(1)
                # PRINTS A DOT TO KNOW WHENEVER A BRANCH IS TAKEN
                # ALSO USEFUL FOR ACKNOWLEDGING THAT THE PROGRAM IS STILL WORKING
                print(".", end='')
    # print(bh)
    # print(len(bh))
    return bh

####################################
### COLLECTS THE BRANCH HISTORY
####################################
def getBranchHistory(f_name, flag):
    retList = []
    if flag == True:
        retList = [0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1]
    else:
        retList = parse(f_name)
    # for i in range(0, 1867817):
    # retList.append(random.randint(0, 1))
    return retList

###################################
### GRABS PREDICTION FOR NAIVE BAYES
###################################
def getPrediction(bh_list, cpt, posts):
    # initialize string that contains most significant bits
    x_zero = ''
    x_one = ''

    i = 0

    for t in bh_list:
        if t == 0:
            # update x_zero and x_one from 0 and 2 positions
            z = cpt[i][0]
            o = cpt[i][2]
            z_bit = getMSB(z)
            x_zero = x_zero + z_bit

            o_bit = getMSB(o)
            x_one = x_one + o_bit

        if t == 1:
            # update x_zero and x_one from 1 and 3 positions
            z = cpt[i][1]
            o = cpt[i][3]

            z_bit = getMSB(z)
            x_zero = x_zero + z_bit

            o_bit = getMSB(o)
            x_one = x_one + o_bit
        i = i + 1
        # end of for

    x_zero_num = int(x_zero)
    x_one_num = int(x_one)

    post_z = int(getMSB(posts[0]))
    post_o = int(getMSB(posts[1]))
    p_zero = x_zero_num + post_z
    p_one = x_one_num + post_o

    #print(p_zero, p_one)
    if p_zero > p_one:
        return 0
    else:
        return 1

#################################
### THIS UPDATES THE CPT TABLE WITH 1S AND 0S WHICH CAN ADD UP TO 15
### IN EACH COLUMN
#################################
def updateCptTable(newVector, table, outcome):
    # If outcome is 0, update p(y=0|x) values
    if outcome == 0:
        i = 0
        for t in newVector:
            # if x = 0, increment 1st column, decrement 2nd column
            if t == 0:
                num0 = table[i][0]
                num1 = table[i][1]

                if num0 < 15:
                    num0 = num0 + 1
                if num1 > 0:
                    num1 = num1 - 1
                table[i][0] = num0
                table[i][1] = num1
            # if x = 1, increment 2nd column, decrement 1st column
            if t == 1:
                num0 = table[i][0]
                num1 = table[i][1]

                if num0 > 0:
                    num0 = num0 - 1
                if num1 < 15:
                    num1 = num1 + 1
                table[i][0] = num0
                table[i][1] = num1

            i = i + 1
    # If outcome is 1, update p(y=1|x) values
    if outcome == 1:
        i = 0
        for t in newVector:
            # if x = 0, increment 3rd column, decrement 4th column
            if t == 0:
                num0 = table[i][2]
                num1 = table[i][3]

                # check for saturation
                if num0 < 15:
                    num0 = num0 + 1
                # check for saturation
                if num1 > 0:
                    num1 = num1 - 1
                table[i][2] = num0
                table[i][3] = num1
            # if x = 1, increment 4th column, decrement 3rd column
            if t == 1:
                num0 = table[i][2]
                num1 = table[i][3]
                # check for saturation
                if num0 > 0:
                    num0 = num0 - 1
                # check for saturation
                if num1 < 15:
                    num1 = num1 + 1
                table[i][2] = num0
                table[i][3] = num1

            i = i + 1
    # print(table)
    return table

#####################################
### GRABS PREDICTION FOR MODIFIED LOGISTIC REGRESSION ALGORITHM
######################################
def getPredLogRegress(bh_list, cpt, posts):

    x_zero = []
    x_one = []

    i = 0

    for t in bh_list:
        if t == 0:
            # update x_zero and x_one from 0 and 2 positions
            z = cpt[i][0]
            o = cpt[i][2]
            z_bit = getMSB(z)
            x_zero.append(int(z_bit))

            o_bit = getMSB(o)
            x_one.append(int(o_bit))

        if t == 1:
            # update x_zero and x_one from 1 and 3 positions
            z = cpt[i][1]
            o = cpt[i][3]

            z_bit = getMSB(z)
            x_zero.append(int(z_bit))
            #print("X_ZERO", x_zero)
            o_bit = getMSB(o)
            x_one.append(int(o_bit))
            #print("X_ONE", x_one)
        i = i + 1
        # end of for
    w = [1 for k in range(len(x_zero))]

    #### PLAY AROUND WITH THIS TO GET DIFFERNT WEIGHTS

    w_ext = [1.5,1.5,1.5,1.5,1.5]

    w = w[:-5]
    for j in w_ext:
        w.append(j)
    w = w[5:]
    w_beg = [0.5,0.5,0.5,0.5,0.5]
    for o in w:
        w_beg.append(o)
    w = w_beg
    w = np.array(w)


    ##### END HERE WEIGHTS

    x_zero = np.array(x_zero)
    x_one = np.array(x_one)
    p_zero = sum(x_zero * w)
    p_one = sum(x_one * w)


    ################################
    ### PLAY AROUND WITH THE FOLLOWING TO GET
    ### A MORE ACCURATE LOG REGRESSION ALGORITHM
    #################################
    '''

    ###THIS IS THE LOGISTIC REGRESSION ALGORITHM
    Xtrain = np.matrix(Xtrain)
    X_T = Xtrain.T

    # The first half of the equation (X^T * X)^1
    XTXI = Xtrain.I * X_T.I

    # print(XTXI.shape)

    # Reformat the shape of the y training set
    Ytrain = Ytrain[:, None]

    # The second half of the equation (X^T*y)
    XTy = X_T * Ytrain
    # print(XTy.shape)

    # Find the matrix w for the training set
    # w = (X^TX)^-1X^Ty
    w = XTXI * XTy

    # Find the transpose of the w matrix for training set A
    transw = w.T

    # Grab the predicted y which is found by w^Tx
    # print(w.T.shape)
    # Multiply the xs of the testing set against the ws add all of those
    # print(Xtest.shape)
    Xtest = np.matrix(Xtest)

    # Reformat the shape of the y testing set
    #Ytest = Ytest[:, None]

    # Predicted y = wT*X
    predY = transw * Xtest.T
    #return predY

    p_zero = x_zero_num + posts[0]'''

    #### RETURNS HIGHEST PROBABILITY
    if p_zero > p_one:
        return 0
    else:
        return 1

###################################
#### COUNTS THE BRANCH HISTORY LIST TO LATER COMPARE WITH
###################################
def getPosteriors(bh_list):
    # Count 0s and 1s
    x = bh_list.count(0)
    y = bh_list.count(1)
    return x, y

###################################
### Test PROOF OF CORRECTNESS
####################################
def getProofOfCorrectness():
    ### BRANCH HISTORY
    retList = [0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1]


    #### IF ALGORITHM IS CORRECT IT SHOULD PREDICT
    shouldbe_list = []
    temp_list = []
    for i in retList:
        temp_list.append(i)
        if len(temp_list) > 30:
            temp_list.pop(0)
        if (sum(temp_list)) >= 15:
            shouldbe_list.append(1)
        else:
            shouldbe_list.append(0)

    #### ACTUAL PREDICTION
    #### CALL FOR ALGORITHM TO WORK
    bh_vector = []
    for x in range(0, BH_Length):
        bh_vector.append(0)

    postProb = (30, 0)

    #  Initialize the CPT Table
    cpt_table = [[0 for x in range(4)] for y in range(BH_Length)]
    for x in range(0, BH_Length):
        cpt_table[x][0] = 0b0000
        cpt_table[x][1] = 0b0000
        cpt_table[x][2] = 0b0000
        cpt_table[x][3] = 0b0000


    l = len(retList)
    predList = []
    i = 0



    # Perform predictions
    while (i < l):
        # get prediction
        pred = getPrediction(bh_vector, cpt_table, postProb)
        predList.append(pred)
        i += 1
    #################################
    ### PRINT DATA FEED, SHOULD BE, PREDICTED BRANCHES
    #################################
    k = 0
    for _ in retList:
        print("Predicted = ", shouldbe_list[k], " --> What it should predict = ",
              predList[k])
        k += 1



###################################
## MAIN FUNCTION
### INITIALIZE CPT TABLE
#### CHECK PROOF OF CORRECTNESS
##### COLLECT BRANCH HISTORY
###### CALL FOR PREDICTION FOR EACH PREDICTOR
####### COLLECT PERFORMANCE METRICS
######## PLOT THE PERFORMANCE METRICS FOR EACH PREDICTOR
#####################################
def main(length_bh, poc):
    bh_vector = []
    for x in range(0, BH_Length):
        bh_vector.append(0)

    postProb = (30, 0)

    #  Initialize the CPT Table
    cpt_table = [[0 for x in range(4)] for y in range(BH_Length)]
    for x in range(0, BH_Length):
        cpt_table[x][0] = 0b0000
        cpt_table[x][1] = 0b0000
        cpt_table[x][2] = 0b0000
        cpt_table[x][3] = 0b0000

    # Get branch history
    bh_history = getBranchHistory("gcc-1K 2.trace",poc)
    print("Branch history:")
    print(len(bh_history))
    l = len(bh_history)
    actualBranchOutcomes = copy.deepcopy(bh_history)
    predList = []
    i = 0

    ### PRED LR = predict log regression (other model)
    predLR = []
    testing = []
    # Perform predictions
    while (i < l):
        # get prediction
        pred = getPrediction(bh_vector, cpt_table, postProb)
        ### Testing LogRegress
        pred2 = getPredLogRegress(bh_vector, cpt_table, postProb)
        predList.append(pred)
        predLR.append(pred2)
        i = i + 1
        #print(i)
        # get actual outcome
        bh_pred = bh_history.pop(0)

        # pass actual outcome to update table
        cpt_table = updateCptTable(bh_vector, cpt_table, bh_pred)

        # update the branch history vector
        bh_vector.pop(0)
        bh_vector.append(bh_pred)

        # update posterior probability
        postProb = getPosteriors(bh_vector)

        testing.append(bh_pred)
    #print("Predictions made")
    #print(len(predList))

    ### PRED T = predict always taken (base model)
    predT = []

    for i in range(l):
        ### THIS IS AN ALWAYS TAKEN BRANCH PREDICTOR
        predT.append(1)

    ### PRED NT = predict never taken (base model)
    predNT = []

    for s in range(l):
        predNT.append(0)

    if poc == True:
        #################################
        ### PRINT DATA FEED, SHOULD BE, PREDICTED BRANCHES
        #################################
        k = 0
        print("TEST RESULTS FOR PROOF OF CORRECTNESS")
        for item in actualBranchOutcomes:

            print("Predicted = ", testing[k], " --> What it should predict = ",
                  actualBranchOutcomes[k])
            k += 1
    else:
    ############################
    ###COUNTERS FOR STATS OF DIFFERENT PREDICTORS
    ############################

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        tp_dummy = 0
        fp_dummy = 0
        tn_dummy = 0
        fn_dummy = 0
        tp_nt = 0
        fp_nt = 0
        tn_nt = 0
        fn_nt = 0
        tp_lr = 0
        fp_lr = 0
        tn_lr = 0
        fn_lr = 0

        # Calculate performance metrics
        # This adds up all of the true/false positives and negatives
        for i in range(0, l):

            if actualBranchOutcomes[i] == 1 and predList[i] == 1:
                tp += 1
            if actualBranchOutcomes[i] == 1 and predList[i] == 0:
                fp += 1
            if actualBranchOutcomes[i] == 0 and predList[i] == 0:
                tn += 1
            if actualBranchOutcomes[i] == 0 and predList[i] == 1:
                fn += 1
            if actualBranchOutcomes[i] == 1 and predLR[i] == 1:
                tp_lr += 1
            if actualBranchOutcomes[i] == 1 and predLR[i] == 0:
                fp_lr += 1
            if actualBranchOutcomes[i] == 0 and predLR[i] == 0:
                tn_lr += 1
            if actualBranchOutcomes[i] == 0 and predLR[i] == 1:
                fn_lr += 1
            if actualBranchOutcomes[i] == 1 and predT[i] == 1:
                tp_dummy += 1
            if actualBranchOutcomes[i] == 1 and predT[i] == 0:
                fp_dummy += 1
            if actualBranchOutcomes[i] == 0 and predT[i] == 0:
                tn_dummy += 1
            if actualBranchOutcomes[i] == 0 and predT[i] == 1:
                fn_dummy += 1
            if actualBranchOutcomes[i] == 1 and predNT[i] == 1:
                tp_nt += 1
            if actualBranchOutcomes[i] == 1 and predNT[i] == 0:
                fp_nt += 1
            if actualBranchOutcomes[i] == 0 and predNT[i] == 0:
                tn_nt += 1
            if actualBranchOutcomes[i] == 0 and predNT[i] == 1:
                fn_nt += 1

        ##########################################
        ### RESULTS FOR NAIVE BAYES PREDICTOR
        ###########################################

        print("\n\n########################################################\n\n")
        accuracy_nb = (tp + tn) / (tp + tn + fp + fn)
        print("PERFORMANCE MEASURES FOR A NAIVE BAYES PREDICTOR WITH BH LENGTH OF ", length_bh)
        print("Accuracy: ", accuracy_nb)
        precision_nb = tp / (tp + fp)
        recall_nb = tp / (tp + fn)
        f1_measure_nb = 2 * (precision_nb * recall_nb) / (precision_nb + recall_nb)
        print("Precision: ", precision_nb)
        print("Recall: ", recall_nb)
        print("F1 Measure: ", f1_measure_nb)
        print("\n\n########################################################\n\n")


        ##########################################
        ### GET RESULTS FOR LOG REG PREDICTOR
        ##########################################

        accuracy_lr = (tp_lr + tn_lr) / (tp_lr + tn_lr + fp_lr + fn_lr)
        print("PERFORMANCE MEASURES FOR A LOGISTIC REGRESSION PREDICTOR WITH BH LENGTH OF ", length_bh)
        print("Accuracy: ", accuracy_lr)
        precision_lr = tp_lr / (tp_lr + fp_lr)
        recall_lr = tp_lr / (tp_lr + fn_lr)
        f1_measure_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)
        print("Precision: ", precision_lr)
        print("Recall: ", recall_lr)
        print("F1 Measure: ", f1_measure_lr)
        print("\n\n########################################################\n\n")



        ##########################################
        ### GET RESULTS FOR ALWAYS TAKEN PREDICTOR
        #########################################

        accuracy_at = (tp_dummy + tn_dummy) / (tp_dummy + tn_dummy + fp_dummy + fn_dummy)
        precision_at = (tp_dummy) / (tp_dummy + fp_dummy)
        recall_at = tp_dummy / (tp_dummy + fn_dummy)
        f1_measure_at = 2 * (precision_at * recall_at) / (precision_at + recall_at)

        print("PERFORMANCE MEASURES FOR AN ALWAYS TAKEN PREDICTOR WITH BH LENGTH OF ", length_bh)
        print("Accuracy: ", accuracy_at)
        print("Precision: ", precision_at)
        print("Recall: ", recall_at)
        print("F1 Measure: ", f1_measure_at)

        print("\n\n########################################################\n\n")

        #########################################
        ### GET RESULTS FOR NEVER TAKEN PREDICTOR
        #########################################

        accuracy_nt = (tp_nt + tn_nt) / (tp_nt + tn_nt + fp_nt + fn_nt)
        precision_nt = (tp_nt) / (tp_nt + fp_nt)
        try:
            recall_nt = tp_nt / (tp_nt + fn_nt)
        except:
            recall_nt = 0.1
        f1_measure_nt = 2 * (precision_nt * recall_nt) / (precision_nt + recall_nt)

        print("PERFORMANCE MEASURES FOR A NEVER TAKEN PREDICTOR WITH BH LENGTH OF ", length_bh)
        print("Accuracy: ", accuracy_nt)
        print("Precision: ", precision_nt)
        print("Recall: ", recall_nt)
        print("F1 Measure: ", f1_measure_nt)

        print("\n\n########################################################\n\n")

        ##############################
        ##### APPEND LIST TO GRAB STATS FOR VARIOUS BH LENGTHS
        ##############################

        accuracy_list_at.append(accuracy_at)
        accuracy_list_nb.append(accuracy_nb)
        accuracy_list_lr.append(accuracy_lr)
        accuracy_list_nt.append(accuracy_nt)
        precision_list_at.append(precision_at)
        precision_list_nb.append(precision_nb)
        precision_list_nt.append(precision_nt)
        precision_list_lr.append(precision_lr)
        recall_list_at.append(recall_at)
        recall_list_nb.append(recall_nb)
        recall_list_nt.append(recall_nt)
        recall_list_lr.append(recall_lr)
        f1_measure_list_at.append(f1_measure_at)
        f1_measure_list_nb.append(f1_measure_nb)
        f1_measure_list_nt.append(f1_measure_nt)
        f1_measure_list_lr.append(f1_measure_lr)

        #####################################
        #### PLOT FOR NAIVE BAYES PREDICTOR
        #####################################

        # x-coordinates of left sides of bars
        left = [0, 50, 100]

        # heights of bars
        height = [accuracy_nb * 100, precision_nb * 100, recall_nb * 100]

        # labels for bars
        tick_label = ['Accuracy', "Precision", 'Recall']

        # plotting a bar chart
        plt.bar(left, height, tick_label=tick_label,
                width=20, color=['red', 'green', 'blue'])

        # naming the x-axis
        plt.xlabel('Performance Metrics')
        # naming the y-axis
        plt.ylabel('Percentage')
        # plot title
        name = "NAIVE BAYES PREDICTOR WITH BH SIZE ", length_bh
        plt.title(name)

        # function to show the plot
        plt.show()

        #####################################
        #### PLOT FOR LOG REGRESS PREDICTOR
        #####################################

        # x-coordinates of left sides of bars
        left = [0, 50, 100]

        # heights of bars
        height = [accuracy_lr * 100, precision_lr * 100, recall_lr * 100]

        # labels for bars
        tick_label = ['Accuracy', "Precision", 'Recall']

        # plotting a bar chart
        plt.bar(left, height, tick_label=tick_label,
                width=20, color=['red', 'green', 'blue'])

        # naming the x-axis
        plt.xlabel('Performance Metrics')
        # naming the y-axis
        plt.ylabel('Percentage')
        # plot title
        name = "LOG REGESSION PREDICTOR WITH BH SIZE ", length_bh
        plt.title(name)

        # function to show the plot
        plt.show()

        ##############################
        ##### PLOT FOR ALWAYS TAKEN PREDICTOR
        ##############################

        # x-coordinates of left sides of bars
        left = [0, 50, 100]

        # heights of bars
        height = [accuracy_at * 100, precision_at * 100, recall_at * 100]

        # labels for bars
        tick_label = ['Accuracy', "Precision", 'Recall']

        # plotting a bar chart
        plt.bar(left, height, tick_label=tick_label,
                width=20, color=['red', 'green', 'blue'])

        # naming the x-axis
        plt.xlabel('Performance Matrix')
        # naming the y-axis
        name = "ALWAYS TAKEN PREDICTOR WITH BH SIZE ", length_bh
        plt.ylabel('Percentage')
        # plot title
        plt.title(name)

        # function to show the plot
        plt.show()

        ###############################
        #### PLOT FOR NEVER TAKEN PREDICTOR
        ###############################

        # x-coordinates of left sides of bars
        left = [0, 50, 100]

        # heights of bars
        height = [accuracy_nt * 100, precision_nt * 100, recall_nt * 100]

        # labels for bars
        tick_label = ['Accuracy', "Precision", 'Recall']

        # plotting a bar chart
        plt.bar(left, height, tick_label=tick_label,
                width=20, color=['red', 'green', 'blue'])

        # naming the x-axis
        plt.xlabel('Performance Matrix')
        # naming the y-axis
        name = "NEVER TAKEN PREDICTOR WITH BH SIZE ", length_bh
        plt.ylabel('Percentage')
        # plot title
        plt.title(name)

        # function to show the plot
        plt.show()


BH_Length = 20

### FOR PROOF OF CORRECTNESS CALL MAIN WITH PoC PARAMETER AS True
main("",True)


#######################################
### STARTS HERE
### RUNS ALGORITHM WITH DIFFERENT SIZED BRANCH HISTORY LENGTHS
########################################
for _ in range(5):
    main(BH_Length, False)

    BH_Length += 5

#########################################
### PLOTS A LINE GRAPH
#########################################
def line_graph(list1, list2, list3, list4, measure):
    ### LINE GRAPH

    # line 1 points
    x1 = [20, 25, 30, 35, 40]
    y1 = [i * 100 for i in list1]

    # plotting the line 1 points
    plt.plot(x1, y1, label="ALWAYS TAKEN")
    plt.ylim(0, 100)
    plt.xlim(20, 40)

    # line 2 points
    x2 = [20, 25, 30, 35, 40]
    y2 = [i * 100 for i in list2]


    # plotting the line 2 points
    plt.plot(x2, y2, label="NAIVE BAYES")

    # line 3 points
    x3 = [20, 25, 30, 35, 40]
    y3 = [p * 100 for p in list3]

    # plotting the line 3 points
    plt.plot(x3, y3, label="NEVER TAKEN")

    # line 4 points
    x4 = [20, 25, 30, 35, 40]
    y4 = [p * 100 for p in list4]

    # plotting the line 4 points
    plt.plot(x4, y4, label="LOGISTIC REGRESSION")

    # naming the x axis
    plt.xlabel('BH SIZE 20, 25, 30, 35, 40')
    # naming the y axis
    plt.ylabel('PERCENTAGE')
    # giving a title to my graph
    title_graph = measure, " WITH DIFFERENT BH SIZE"
    plt.title(title_graph)

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()


########################################
### CALLS TO PLOT A PARTICULAR PERFORMANCE METRIC FOR EACH OF THE PREDICTORS
########################################
line_graph(accuracy_list_at, accuracy_list_nb, accuracy_list_nt, accuracy_list_lr, "ACCURACY")
line_graph(precision_list_at, precision_list_nb, precision_list_nt, precision_list_lr, "PRECISION")
line_graph(recall_list_at, recall_list_nb, recall_list_nt, recall_list_lr,"RECALL")
line_graph(f1_measure_list_at, f1_measure_list_nb, f1_measure_list_nt, f1_measure_list_lr, "F1-SCORE")
