import json
import joblib
import numpy as np
from azureml.core.model import Model
from nltk.tokenize import WhitespaceTokenizer

# Features

# Username specific features
def LowerUnderscoreUpper(s):
    l = len(s)

    for i in range(0,l-2):
        if s[i].islower() and s[i+1] == '_' and s[i+2].isupper():
            return True
    return False

def HasUnderscore(s):
    s = str(s)

    for c in s:
        if c == '_':
            return True
    return False

def LowerUpperLower(s):
    s = str(s)

    for i in range(0, len(s)-2):
        if s[i].islower() and s[i+1].isupper() and s[i+2].islower():
            return True
    return False

def MultipleLowerUpperLower(s):
    s = str(s)

    flag = False
    for i in range(0, len(s)-2):
        if s[i].islower() and s[i+1].isupper() and s[i+2].islower():
            if flag:
                return True
            else:
                flag = True
    return False

def ExactlyTwoUppercase(s):
    count = 0

    for c in s:
        if c.isupper():
            count+=1
    if count ==2:
        return True
    else:
        return False

def AllLowerMoreThan(bound, s):
    s = str(s)

    if len(s) < bound:
        return False
    for c in s:
        if c.isupper():
            return False
    return True

def AdjacentUppers(s):
    s = str(s)

    for i in range(0,len(s)-1):
        if s[i].isupper() and s[i+1].isupper():
            return True
    return False

def StartLetterEndNonLetter(s):
    s = str(s)

    if s[0].isalpha() and not s[-1].isalpha():
        return True
    return False


# Hostname specific features
def LengthGTLT(gt, ls, s):
    l = len(str(s))
    if l >= gt and l <= ls:
        return True
    else:
        return False

def IllegalHostnameChars(s):
    s = str(s)
    illegal_chars = [".", "\", ""/", "*", "?", "\"", "<", ">", "|", ",", "~", ":", "!", "@", "#", "$", "%", "^", "&", "'", "(", ")", "{", "}", " "]
    for c in s:
        if c in illegal_chars:
            return True
    return False

def AlphaOrDigit(s):
    s = str(s)
    if s.isalpha() or s.isdigit():
        return True
    return False

def HostIllegalEnding(s):
    s = str(s)
    if s[-1] == '-' or s[-1] == '.':
        return True
    return False

# Filename specific features

def ContainsPeriod(s):
    s = str(s)
    for c in s:
        if c == '.':
            return True
    return False

def IsUrl(s):
    s = str(s)
    l = len(s)
    if l > 3:
        if s[0:4] == "http":
            return True
    if l > 4:
        if s[0:5] == "https":
            return True
    return False

def HasSlash(s):
    s = str(s)
    for c in s:
        if c == '/' or c == '\\':
            return True
    return False

def HasMultipleSlash(s):
    s = str(s)
    slashes = ['/','\\']
    flag = False
    for c in s:
        if not flag and c in slashes:
            flag = True
        elif flag and c in slashes:
            return True
    return False

def HasPossibleExtension(s):
    s = str(s)
    for i in range(0,len(s)):
        if s[i] == '.':
            x = s[i:-1]
            if len(x) >= 2 and len(x) <= 4:
                return True
    return False

# IP Features
def NumbersThenPeriod(s):
    s = str(s)
    freq = 0
    for i in range(0,len(s)-1):
        if s[i] >= '0' and s[i] <= '9' and s[i+1] == '.':
            freq += 1
    return True if freq == 3 else False

def AtLeastFourDigits(s):
    s = str(s)
    counter = 0
    for c in s:
        if c.isnumeric():
            counter += 1
    return True if counter >=4 else False

# Servername Features

def HasPeriodAndSlash(s):
    s = str(s)
    period = False
    slash = False
    for c in s:
        if c == '.':
            period = True
        if s =='/':
            slash = True
    return True if period and slash else False

def HasInstanceNumPeriod(s):
    s = str(s)
    for i in range(0,len(s)-1):
        if s[i].isnumeric() and s[i+1] == '.':
            return True
    return False

def HasMultipleNumPeriod(s):
    s = str(s)
    flag = False
    for i in range(0,len(s)-1):
        if s[i].isnumeric() and s[i+1] == '.' and not flag:
            flag = True
        if s[i].isnumeric() and s[i+1] == '.' and flag:
            return True
    return False

def ExtractFeatures(s):
    tk = WhitespaceTokenizer() 

    tokens = tk.tokenize(s)
    tokens = list(filter(lambda x: not IsUrl(x), tokens))
    dp_list = []
    dp_list.append(True) if True in list(map(lambda x: LowerUnderscoreUpper(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: HasUnderscore(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: LowerUpperLower(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: MultipleLowerUpperLower(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: ExactlyTwoUppercase(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: AllLowerMoreThan(10,x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: AdjacentUppers(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: StartLetterEndNonLetter(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: LengthGTLT(1, 15, x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: IllegalHostnameChars(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: AlphaOrDigit(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: HostIllegalEnding(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: ContainsPeriod(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: HasSlash(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: HasMultipleSlash(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: HasPossibleExtension(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: NumbersThenPeriod(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: AtLeastFourDigits(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: HasPeriodAndSlash(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: HasInstanceNumPeriod(x), tokens)) else dp_list.append(False)
    dp_list.append(True) if True in list(map(lambda x: HasMultipleNumPeriod(x), tokens)) else dp_list.append(False)

    return dp_list

    
# Called when the service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path('TestModel1')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    string = json.loads(raw_data)['data']

    data = np.array(ExtractFeatures(string))
    # Get a prediction from the model
    predictions = model.predict(data.reshape(1,-1))
    # Return the predictions as any JSON serializable format
    return predictions.tolist()