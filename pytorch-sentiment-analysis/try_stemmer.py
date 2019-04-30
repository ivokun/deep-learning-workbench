from nltk.stem import PorterStemmer

plurals = ['caresses', 'flies', 'dies', 'mules', 'denied',
           'died', 'agreed', 'owned', 'humbled', 'sized',
           'meeting', 'stating', 'siezing', 'itemization',
           'sensational', 'traditional', 'reference', 'colonizer',
           'plotted']
ps = PorterStemmer()
stem = lambda datas: [ps.stem(x) for x in datas] + datas

print(stem(plurals))