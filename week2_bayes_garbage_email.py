
# coding: utf-8

# ### 贝叶斯分类模型 垃圾邮件分类

# In[58]:


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  # 词集模型
        else:
            print("the word: {:s} is not in my Bocabulary!".fotmat(word))
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] =+ 1  # 词袋模型
        else:
            print("the word: {:s} is not in my Bocabulary!".fotmat(word))
    return returnVec


# In[53]:

def trainNB0(trainMatrix, trainCategory):
    """
    input: 训练集，训练集标签列表
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    trainCategory_1_num = sum([i for i in trainCategory if int(i)==1])
    pAbusive = trainCategory_1_num/float(numTrainDocs)
    p0Num = numpy.zeros(numWords); p1Num = numpy.zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
#             print(trainMatrix[i])
            p1Num += trainMatrix[i]
#             print(p1Num)
            p1Denom += sum(trainMatrix[i])
#             print(p1Denom)
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0


# In[56]:

def textParse(bigString):    #input is big string, #output is word list
    import re
    import random
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest():
    import random
    import numpy
    from numpy import array
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('example_data/ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('example_data/ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    
    
#python3.x , 出现错误 'range' object doesn't support item deletion

#原因：python3.x   range返回的是range对象，不返回数组对象

#解决方法：把 trainingSet = range(50) 改为 trainingSet = list(range(50))


# In[57]:

spamTest()


# ### 从个人广告中获取区域倾向

# In[ ]:




# In[ ]:



