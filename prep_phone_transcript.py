import numpy as np
import csv
import sys

with open(sys.argv[1]+'/lexicon.txt','r') as csv_file:
        lexicon = list(csv.reader(csv_file,delimiter='\t'))
        #lexicon=lexicon[0:3870] #upto only english characters

with open(sys.argv[2],'r') as csv_file:
        words = list(csv.reader(csv_file,delimiter=' '))

tr=np.copy(words)
for i in range(len(words)):
        for j in range(len(words[i])):
                for phone in lexicon:
                        if(words[i][j]==phone[0]):
                                tr[i][j]=phone[1]
                                #tr[i][-1]='</s>'
with open(sys.argv[2]+'_phn.txt','w') as csv_file2:
        writer=csv.writer(csv_file2,delimiter=' ',quotechar =" ")
        writer.writerows(tr)

        ##run this 2-3 times in bash: sed -i 's/  / /g' trans_for_phone_LM.txt

