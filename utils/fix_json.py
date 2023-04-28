import json

def singleQuoteToDoubleQuote(singleQuoted):
            cList=list(singleQuoted)
            inDouble=False
            inBracket=False
            inSingle=False
            for i,c in enumerate(cList):
                if c=="'":
                    if not inDouble and not inBracket:
                        inSingle = not inSingle
                        cList[i] = '"'
                elif c=='"':
                    inDouble = not inDouble
                elif c=='(' or c==')':
                    inBracket = not inBracket
            doubleQuoted = "".join(cList)    
            return doubleQuoted


file_object = open('test.json', 'r')
out_string = file_object.read()
file_object.close()
out_string = singleQuoteToDoubleQuote(out_string)
out_file = open('results2.json', 'w')
n = out_file.write(out_string)
out_file.close()