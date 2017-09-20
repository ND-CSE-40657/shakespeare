import sys

if sys.version_info < (3,0,0):
    raise ImportError("Python 3 or higher required.")

def levenshtein(s, t):
    """http://en.wikipedia.org/wiki/Levenshtein_distance"""
    m = len(s)
    n = len(t)
    # d[i][j] says how to get t[:j] from s[:i]
    d = [[None for j in range(n+1)] for i in range(m+1)]
    d[0][0] = 0
    for i in range(m+1):
        for j in range(n+1):
            if i == j == 0: continue
            
            cands = []
            
            if i > 0 and j > 0 and  d[i-1][j-1] is not None:
                antcost = d[i-1][j-1]
                if s[i-1] == t[j-1]:
                    cands.append(antcost+0)
                else:
                    cands.append(antcost+1)

            if i > 0 and d[i-1][j] is not None:
                # deletion
                antcost = d[i-1][j]
                cands.append(antcost+1)

            if j > 0 and d[i][j-1] is not None:
                # insertion
                antcost = d[i][j-1]
                cands.append(antcost+1)

            d[i][j] = min(cands)
    return float(d[m][n])

def cer(data):
    """Compute character error rate. The argument data should be a list of
    pairs of strings. In each pair, the first string is the correct string
    and the second string is the string to be tested."""
    d = 0
    n = 0
    for (r, t) in data:
        d += levenshtein(r, t)
        n += len(r)
    return d/n

if __name__ == "__main__":
    try:
        reffilename, testfilename = sys.argv[1:]
    except ValueError():
        sys.stderr.write("usage: cer.py test.new youroutput.new")
        sys.exit(1)
        
    ref = list(open(reffilename))
    test = list(open(testfilename))
    if len(ref) != len(test):
        sys.stderr.write("error: files are not the same length")
        sys.exit(1)

    print(cer(zip(ref, test)))

    
