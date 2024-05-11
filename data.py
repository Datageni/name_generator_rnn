# [Libraries]{Data Preprocessing}
from io import open 
import glob
import os
import unicodedata
import string


all_letters = string.ascii_letters + ".,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

# [Function]{File Finding}
def findFiles(path : string): 
    """
    Searchs for files given an specific path 
    """
    return glob.glob(path)

# [Function]{String Convertion}
def unicodeToAscii(s : string):
    """
    Converts unicode strings into plain ASCII strings
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# [Function]{File Reading}
def readLines(filename):
    """
    Reads files and split's them into lines
    """
    with open(filename, encoding='utf-8') as some_file:
        return[unicodeToAscii(line.strip()) for line in some_file]


# [Dictonary]{List Of Lines Per Category}
category_lines = {}
# [Array]{Stores All Categories}
all_categories = []

# Looping in all files of the data folder in  the names folder that ends with '.txt'
for filename in findFiles('data/names/*.txt'):
    # Spliting filename path and retrieving the first element of the file name
    category = os.path.splitext(os.path.basename(filename))[0]
    # Appending categories to all_categories array
    all_categories.append(category)
    # Reading file and spliting it into lines
    lines = readLines(filename)
    # Filling the category_lines dictionary with lines and it specific category
    category_lines[category] = lines
# Retrieving the number of categories 
n_categories = len(all_categories) 

# Error handling when data is equal to 0
if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data')

# Printing the number of categories and all of them
print('# categories: ', n_categories, all_categories)

