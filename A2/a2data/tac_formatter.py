import sys, os, re, string

f = open(sys.argv[1])
content = f.read()
content = content.lower()
alphabet = ''.join(list(map(chr, range(97, 123))) + ['.',',',' ','\r','\n'])
content = [c for c in content if c in alphabet]
i=0
while i != len(content)-1:
	print str(i) + "/" + str(len(content))
	if content[i] == ' ' and content[i-1] == ' ':
		del content[i-1]
	i+=1
content = ''.join(content)
text_file = open(sys.argv[2], "w")
text_file.write(content)
text_file.close()