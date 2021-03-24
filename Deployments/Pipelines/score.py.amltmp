import json
import joblib
import numpy as np
from azureml.core.model import Model
import nltk
from nltk.tokenize import WhitespaceTokenizer
from ipaddress import ip_address
nltk.download('averaged_perceptron_tagger')

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


def PosTag(s):
    d = {'CC':1,
        'CD': 2,
        'DT': 3,
        'EX': 4,
        'FW': 5,
        'IN': 6,
        'JJ': 7,
        'JJR': 8,
        'JJS': 9,
        'LS': 10,
        'MD': 11,
        'NN': 12,
        'NNS': 13,
        'NNP': 14,
        'NNPS': 15,
        'PDT': 16,
        'POS': 17,
        'PRP': 18,
        'PRP$': 19,
        'RB': 20,
        'RBR': 21,
        'RBS': 22,
        'RP': 23,
        'TO': 24,
        'UH': 25,
        'VB': 26,
        'VBD': 27,
        'VBG': 28,
        'VBN': 29,
        'VBP': 30,
        'VBZ': 31,
        'WDT': 32,
        'WP': 33,
        'WP$': 34,
        'WRB': 35
        }
    tagged = nltk.pos_tag([s,"test"])
    return d[tagged[0][1]] if tagged[0][1] in d else 0


def ExtractFeatures(x):
    feature_list = []
    feature_list.append(LowerUnderscoreUpper(x))
    feature_list.append(HasUnderscore(x))
    feature_list.append(LowerUpperLower(x))
    feature_list.append(MultipleLowerUpperLower(x))
    feature_list.append(ExactlyTwoUppercase(x))
    feature_list.append(AllLowerMoreThan(10,x))
    feature_list.append(AdjacentUppers(x))
    feature_list.append(StartLetterEndNonLetter(x))
    feature_list.append(LengthGTLT(1, 15, x))
    feature_list.append(IllegalHostnameChars(x))
    feature_list.append(AlphaOrDigit(x))
    feature_list.append(HostIllegalEnding(x))
    feature_list.append(ContainsPeriod(x))
    feature_list.append(HasSlash(x))
    feature_list.append(HasMultipleSlash(x))
    feature_list.append(HasPossibleExtension(x))
    feature_list.append(NumbersThenPeriod(x))
    feature_list.append(AtLeastFourDigits(x))
    feature_list.append(HasPeriodAndSlash(x))
    feature_list.append(HasInstanceNumPeriod(x))
    feature_list.append(HasMultipleNumPeriod(x))
    feature_list.append(PosTag(x))
    return feature_list

    
# Called when the service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path('ModelV2')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    string = json.loads(raw_data)['data']
    tk = WhitespaceTokenizer()
    tokens = tk.tokenize(string)
    not_ip = []
    labels = []
    found_attributes = {}

    for token in string.split():
        try:                
            ip = ip_address(token)
            found_attributes[token] = "IP"
        except:
            not_ip.append(token)
    for i,point in enumerate(not_ip):
        features = ExtractFeatures(point)
        labels.append((i,model.predict([features])))
    for label in labels:
        if label[1][0] != 'nothing':
            found_attributes[not_ip[label[0]]] = label[1][0]

    # Return the predictions as any JSON serializable format
    return found_attributes